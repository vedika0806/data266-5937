#!/usr/bin/env python
"""HW4.a / HW4.5 BERT fine-tuning runner.

One configuration per process, so peak-memory statistics start clean and an
out-of-memory error ends only this run (the notebook records it and moves on).
Writes one JSON result file to --out.

  --mode full   : train for --epochs with per-epoch validation, then test (HW4.a, Part B accuracy runs)
  --mode bench  : warm-up steps, timed steps on random batches, then one real epoch (Part A/B sweeps)
"""
import argparse, json, math, os, random, sys, time, traceback

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

p = argparse.ArgumentParser()
p.add_argument("--split", default="data/split.pt")
p.add_argument("--model", default="bert-base-uncased")
p.add_argument("--mode", choices=["full", "bench"], default="full")
p.add_argument("--batch", type=int, default=32)            # physical (micro) batch
p.add_argument("--accum", type=int, default=1)             # gradient-accumulation steps
p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
p.add_argument("--grad_ckpt", type=int, default=0)
p.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--epochs", type=int, default=4)
p.add_argument("--warmup_frac", type=float, default=0.1)
p.add_argument("--weight_decay", type=float, default=0.01)
p.add_argument("--seed", type=int, default=5937)
p.add_argument("--eval_batch", type=int, default=128)
p.add_argument("--bench_warmup", type=int, default=3)
p.add_argument("--bench_steps", type=int, default=20)
p.add_argument("--bench_epoch", type=int, default=1)      # also time one real epoch in bench mode
p.add_argument("--max_steps", type=int, default=0)        # full mode: stop after N optimizer steps (0 = off)
p.add_argument("--save_model", default="")
p.add_argument("--save_preds", default="")
p.add_argument("--device", default="cuda")
p.add_argument("--tag", default="")
p.add_argument("--out", required=True)
a = p.parse_args()

result = {"tag": a.tag, "mode": a.mode, "config": vars(a).copy(), "ok": False, "oom": False}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def is_oom(e):
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    m = str(e).lower()
    return "out of memory" in m or "cublas_status_alloc_failed" in m


def write_result():
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)


def main():
    from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup

    set_seed(a.seed)
    dev = torch.device("cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu")
    cuda = dev.type == "cuda"
    result["device"] = str(dev)
    result["torch"] = torch.__version__
    if cuda:
        result["gpu"] = torch.cuda.get_device_name(0)
        result["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    # ---- data: the exact split built in the notebook (same rows, same SEED) ----
    d = torch.load(a.split, weights_only=False)
    classes = list(d["classes"])
    def tds(k):
        return TensorDataset(d[k]["input_ids"], d[k]["attention_mask"], d[k]["labels"])
    tr, va, te = tds("train"), tds("val"), tds("test")
    n_train = len(tr)
    result.update(n_train=n_train, n_val=len(va), n_test=len(te), classes=classes,
                  max_len=int(d["train"]["input_ids"].shape[1]))

    # ---- model ----
    model = BertForSequenceClassification.from_pretrained(
        a.model, num_labels=len(classes),
        id2label={i: c for i, c in enumerate(classes)}, label2id={c: i for i, c in enumerate(classes)})
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(dev)
    result["n_params"] = sum(p.numel() for p in model.parameters())

    # ---- optimizer ----
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": a.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    if a.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, lr=a.lr)
    else:
        opt = torch.optim.AdamW(groups, lr=a.lr)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(a.amp)
    def autocast():
        return torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None)
    scaler = torch.amp.GradScaler(dev.type, enabled=(a.amp == "fp16" and cuda))

    g = torch.Generator().manual_seed(a.seed)
    train_loader = DataLoader(tr, batch_size=a.batch, shuffle=True, generator=g, pin_memory=cuda)
    val_loader = DataLoader(va, batch_size=a.eval_batch)
    test_loader = DataLoader(te, batch_size=a.eval_batch)

    micro_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(micro_per_epoch / a.accum)
    if a.mode == "full":
        total_steps = steps_per_epoch * a.epochs
    else:
        total_steps = a.bench_warmup + a.bench_steps + steps_per_epoch
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup_frac * total_steps), total_steps)
    result.update(effective_batch=a.batch * a.accum, steps_per_epoch=steps_per_epoch)

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def peak():
        if not cuda:
            return 0.0, 0.0
        return torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9

    def micro(batch):
        ids, mask, y = (t.to(dev, non_blocking=True) for t in batch)
        with autocast():
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
        scaler.scale(loss / a.accum).backward()
        return loss.detach().float(), len(y)

    def opt_step():
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        tot, n, preds, probs, ys = 0.0, 0, [], [], []
        for batch in loader:
            ids, mask, y = (t.to(dev) for t in batch)
            with autocast():
                out = model(input_ids=ids, attention_mask=mask, labels=y)
            tot += out.loss.float().item() * len(y)
            n += len(y)
            pr = out.logits.float().softmax(-1).cpu()
            probs.append(pr); preds.append(pr.argmax(-1)); ys.append(y.cpu())
        model.train()
        preds, ys, probs = torch.cat(preds), torch.cat(ys), torch.cat(probs)
        return tot / n, (preds == ys).float().mean().item(), preds.numpy(), probs.numpy(), ys.numpy()

    model.train()
    if a.mode == "bench":
        X_ids, X_mask, Y = tr.tensors
        def rand_step():
            for _ in range(a.accum):
                idx = torch.randint(0, n_train, (a.batch,), generator=g)
                micro((X_ids[idx], X_mask[idx], Y[idx]))
            opt_step()

        for _ in range(a.bench_warmup):
            rand_step()
        sync(); t0 = time.perf_counter()
        for _ in range(a.bench_steps):
            rand_step()
        sync(); dt = time.perf_counter() - t0
        thr = a.bench_steps * a.batch * a.accum / dt
        result.update(throughput=thr, step_time_s=dt / a.bench_steps, epoch_time_est_s=n_train / thr)
        if a.bench_epoch:
            sync(); t1 = time.perf_counter()
            for i, b in enumerate(train_loader):
                micro(b)
                if (i + 1) % a.accum == 0 or i + 1 == micro_per_epoch:
                    opt_step()
            sync()
            result["epoch_time_s"] = time.perf_counter() - t1
        pa, pr = peak()
        result.update(peak_alloc_gb=pa, peak_reserved_gb=pr)
        return

    # ---------------- full training ----------------
    history, best_val, best_epoch, best_state = [], float("inf"), 0, None
    opt_steps, train_time, peak_a, peak_r, samples_seen = 0, 0.0, 0.0, 0.0, 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    stop = False
    wall0 = time.perf_counter()
    for ep in range(1, a.epochs + 1):
        model.train(); sync(); t0 = time.perf_counter()
        loss_sum, nmb, nsamp = torch.zeros((), device=dev), 0, 0
        for i, b in enumerate(train_loader):
            l, bs = micro(b)
            loss_sum += l; nmb += 1; nsamp += bs
            if (i + 1) % a.accum == 0 or i + 1 == micro_per_epoch:
                opt_step(); opt_steps += 1
                if a.max_steps and opt_steps >= a.max_steps:
                    stop = True
                    break
        sync(); et = time.perf_counter() - t0
        train_time += et; samples_seen += nsamp
        pa, pr = peak(); peak_a, peak_r = max(peak_a, pa), max(peak_r, pr)
        vl, vacc, _, _, _ = evaluate(val_loader)
        if cuda:
            torch.cuda.reset_peak_memory_stats()     # keep eval out of the training peak
        rec = dict(epoch=ep, train_loss=(loss_sum / nmb).item(), val_loss=vl, val_acc=vacc,
                   epoch_time_s=et, samples_per_s=nsamp / et, optimizer_steps=opt_steps)
        history.append(rec)
        print(json.dumps(rec), flush=True)
        if vl < best_val:
            best_val, best_epoch = vl, ep
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if stop:
            break
    wall = time.perf_counter() - wall0

    tl_f, ta_f, pred_f, prob_f, y_true = evaluate(test_loader)          # model after the last epoch
    model.load_state_dict(best_state)
    tl_b, ta_b, pred_b, prob_b, _ = evaluate(test_loader)               # best-validation checkpoint

    result.update(
        history=history, best_epoch=best_epoch, best_val_loss=best_val,
        last_epoch=history[-1]["epoch"], epochs_past_best=history[-1]["epoch"] - best_epoch,
        test_loss_final=tl_f, test_acc_final=ta_f, test_loss_best=tl_b, test_acc_best=ta_b,
        train_time_s=train_time, wall_time_s=wall, throughput=samples_seen / train_time,
        optimizer_steps=opt_steps, peak_alloc_gb=peak_a, peak_reserved_gb=peak_r)
    if a.save_preds:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_preds)), exist_ok=True)
        np.savez(a.save_preds, y_true=y_true, pred_final=pred_f, prob_final=prob_f,
                 pred_best=pred_b, prob_best=prob_b)
    if a.save_model:
        model.save_pretrained(a.save_model)                              # best-validation weights


try:
    main()
    result["ok"] = True
except Exception as e:  # noqa: BLE001
    if is_oom(e):
        result["oom"] = True
        result["error"] = "OOM: " + str(e)[:300]
    else:
        result["error"] = traceback.format_exc()[-3000:]
        print(result["error"], file=sys.stderr)
finally:
    write_result()
sys.exit(0 if (result["ok"] or result["oom"]) else 1)
