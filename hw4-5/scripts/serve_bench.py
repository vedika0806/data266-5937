#!/usr/bin/env python
"""Serving benchmarks for HW4.a and HW4.5 Part D. One mode per process.

  hf       plain Hugging Face text-generation pipeline, batch 1 (the baseline)
  vllm     vLLM offline engine at --batches
  vllm_kv  start vLLM with a long max_model_len and record its KV-cache capacity,
           then check that --kv_batch requests of the implied length really run
  kv       KV-cache memory vs context length at a fixed batch (HF, chunked prefill
           until the card runs out of memory)

Method (same for HF and vLLM): greedy decoding, every request generates exactly
--max_new_tokens tokens (min_new_tokens / ignore_eos), so tokens/s are comparable.
A request's latency is the wall time of the batch it was submitted in, because an
offline batch returns all its requests together.
"""
import argparse, json, math, os, sys, time, traceback

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["hf", "vllm", "kv", "vllm_kv"], required=True)
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--dtype", default="bfloat16")
p.add_argument("--batches", default="1,8,32")
p.add_argument("--max_new_tokens", type=int, default=128)
p.add_argument("--n_req", type=int, default=64)
p.add_argument("--min_rounds", type=int, default=10)
p.add_argument("--warmup", type=int, default=2)
p.add_argument("--gpu_mem_util", type=float, default=0.90)
p.add_argument("--max_model_len", type=int, default=4096)
p.add_argument("--kv_batch", type=int, default=8)
p.add_argument("--kv_chunk", type=int, default=1024)
p.add_argument("--kv_max", type=int, default=262144)
p.add_argument("--seed", type=int, default=5937)
p.add_argument("--tag", default="")
p.add_argument("--out", required=True)
a = p.parse_args() if __name__ == "__main__" else None

# Own prompts ("experiment with your own prompt"). Edit freely; keep the list fixed between HW4.a and Part D.
PROMPTS = [
    "Explain in three sentences why airline customer-service tweets are hard to classify by sentiment.",
    "Write a short, polite reply to a customer whose flight was delayed by four hours.",
    "Summarise the difference between precision and recall for a non-technical manager.",
    "Give three tips for writing a clear bug report.",
    "Describe how a GPU differs from a CPU in one paragraph.",
    "Write a haiku-free, two-sentence product description for a reusable water bottle.",
    "What are the pros and cons of fine-tuning BERT versus prompting a large language model for sentiment analysis?",
    "List four common causes of out-of-memory errors when training neural networks.",
    "Explain what a KV cache is in a transformer, as if to a first-year student.",
    "Draft a tweet announcing a new campus coffee shop, with a friendly tone.",
    "Why does larger batch size increase throughput but not always reduce training time to a target accuracy?",
    "Translate the idea of 'technical debt' into an analogy about home maintenance.",
    "Give a step-by-step plan to study for a machine-learning final exam in one week.",
    "What is mixed-precision training and why does it save memory?",
    "Write a short motivational message for a student debugging code at 2 a.m.",
    "Compare trains and planes for a 500-mile trip in a few sentences.",
]

result = {"tag": a.tag, "mode": a.mode, "config": vars(a).copy(), "ok": False, "oom": False} if a else {}

def is_oom(e):
    try:
        import torch
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    m = str(e).lower()
    return "out of memory" in m or "cublas_status_alloc_failed" in m


def pct(x, q):
    return float(np.percentile(np.asarray(x, dtype=float), q))


def row(name, B, req_lat, gen_tokens, total_time):
    return dict(configuration=name, batch=B, requests=len(req_lat),
                tokens_per_s=gen_tokens / total_time,
                p50_ms=pct(req_lat, 50) * 1000, p95_ms=pct(req_lat, 95) * 1000,
                mean_ms=float(np.mean(req_lat)) * 1000,
                requests_per_s=len(req_lat) / total_time,
                total_tokens=int(gen_tokens), total_time_s=total_time)


def chat_prompts(n):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    out = []
    for i in range(n):
        text = PROMPTS[i % len(PROMPTS)]
        if getattr(tok, "chat_template", None):
            out.append(tok.apply_chat_template([{"role": "user", "content": text}],
                                               tokenize=False, add_generation_prompt=True))
        else:
            out.append(text)
    return tok, out


def run_hf():
    import torch
    from transformers import AutoModelForCausalLM, pipeline
    n = max(a.n_req, a.min_rounds)
    tok, prompts = chat_prompts(n + a.warmup)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to("cuda").eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tok, device=0)
    gk = dict(max_new_tokens=a.max_new_tokens, min_new_tokens=a.max_new_tokens, do_sample=False,
              return_full_text=False,
              pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    for i in range(a.warmup):
        pipe(prompts[i], **gk)
    lats, retok, samples = [], [], []
    for i in range(n):
        t0 = time.perf_counter()
        o = pipe(prompts[a.warmup + i], **gk)
        lats.append(time.perf_counter() - t0)
        text = o[0]["generated_text"]
        retok.append(len(tok(text, add_special_tokens=False).input_ids))
        if len(samples) < 3:
            samples.append({"prompt": PROMPTS[(a.warmup + i) % len(PROMPTS)], "output": text})
    gen_tokens = a.max_new_tokens * n          # exact: min_new_tokens == max_new_tokens
    result["rows"] = [row("HF pipeline, batch 1", 1, lats, gen_tokens, sum(lats))]
    result["retokenized_mean_tokens"] = float(np.mean(retok))
    result["samples"] = samples
    result["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9


def make_llm(max_model_len):
    from vllm import LLM
    return LLM(model=a.model, dtype=a.dtype, gpu_memory_utilization=a.gpu_mem_util,
               max_model_len=max_model_len, seed=a.seed, enable_prefix_caching=False)


def kv_capacity(llm):
    """Best effort: read the KV-cache block count from the engine (the notebook also parses the log)."""
    for path in ("llm_engine.vllm_config.cache_config", "llm_engine.cache_config"):
        obj = llm
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            blocks, bs = getattr(obj, "num_gpu_blocks", None), getattr(obj, "block_size", None)
            if blocks and bs:
                return int(blocks) * int(bs)
        except AttributeError:
            continue
    return None


def run_vllm():
    from vllm import SamplingParams
    import vllm
    result["vllm_version"] = vllm.__version__
    batches = [int(b) for b in a.batches.split(",")]
    tok, prompts = chat_prompts(max(512, max(batches) * 4))
    llm = make_llm(a.max_model_len)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, ignore_eos=True, seed=a.seed)
    rows, samples, k = [], [], 0
    for B in batches:
        for _ in range(a.warmup):
            llm.generate(prompts[:B], sp, use_tqdm=False)
        rounds = max(a.min_rounds, math.ceil(a.n_req / B))
        req_lat, tot_tok, tot_t = [], 0, 0.0
        for _ in range(rounds):
            batch = [prompts[(k + j) % len(prompts)] for j in range(B)]
            k += B
            t0 = time.perf_counter()
            outs = llm.generate(batch, sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            tot_t += dt
            tot_tok += sum(len(o.outputs[0].token_ids) for o in outs)
            req_lat += [dt] * B
            if B == batches[0] and len(samples) < 3:
                samples.append({"prompt": PROMPTS[(k - B) % len(PROMPTS)], "output": outs[0].outputs[0].text})
        rows.append(row(f"vLLM, batch {B}", B, req_lat, tot_tok, tot_t))
        print(json.dumps(rows[-1]), flush=True)
    result["rows"] = rows
    result["samples"] = samples
    result["kv_capacity_tokens"] = kv_capacity(llm)


def run_vllm_kv():
    from vllm import SamplingParams
    import vllm
    result["vllm_version"] = vllm.__version__
    llm = make_llm(a.max_model_len)
    cap = kv_capacity(llm)
    result["kv_capacity_tokens"] = cap
    if cap:
        B = a.kv_batch
        ctx = min(cap // B, a.max_model_len)
        P = ctx - 32 - 8
        rng = np.random.default_rng(a.seed)
        reqs = [{"prompt_token_ids": rng.integers(1000, 20000, size=P).tolist()} for _ in range(B)]
        sp = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate(reqs, sp, use_tqdm=False)
        result["check"] = dict(batch=B, prompt_tokens=P, context_tokens=P + 32,
                               seconds=time.perf_counter() - t0,
                               completed=sum(len(o.outputs[0].token_ids) == 32 for o in outs))


def run_kv():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype),
                                                 attn_implementation="sdpa").to("cuda").eval()
    layers = cfg.num_hidden_layers
    kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    elem = torch.finfo(getattr(torch, a.dtype)).bits // 8
    per_tok = 2 * layers * kv_heads * head_dim * elem          # K and V, every layer
    B, chunk = a.kv_batch, a.kv_chunk
    result.update(layers=layers, kv_heads=kv_heads, head_dim=head_dim, bytes_per_elem=elem,
                  kv_bytes_per_token=per_tok, batch=B, chunk=chunk,
                  max_position_embeddings=getattr(cfg, "max_position_embeddings", None),
                  gpu_total_gb=torch.cuda.get_device_properties(0).total_memory / 1e9)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    result["weights_gb"] = base / 1e9
    g = torch.Generator(device="cuda").manual_seed(a.seed)
    points, cache, L = [], None, 0
    result["points"] = points
    try:
        with torch.no_grad():
            while L + chunk <= a.kv_max:
                ids = torch.randint(1000, min(cfg.vocab_size, 20000), (B, chunk), device="cuda", generator=g)
                out = model.model(input_ids=ids, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                del out
                L += chunk
                torch.cuda.synchronize()
                points.append(dict(context=L,
                                   kv_measured_gb=(torch.cuda.memory_allocated() - base) / 1e9,
                                   kv_theory_gb=B * L * per_tok / 1e9,
                                   peak_alloc_gb=torch.cuda.max_memory_allocated() / 1e9))
        result["stopped"] = "kv_max reached"
    except Exception as e:  # noqa: BLE001
        if not is_oom(e):
            raise
        result["stopped"] = f"OOM while extending to {L + chunk} tokens"
    result["max_context_ok"] = L


if __name__ == "__main__":
    try:
        {"hf": run_hf, "vllm": run_vllm, "kv": run_kv, "vllm_kv": run_vllm_kv}[a.mode]()
        result["ok"] = True
    except Exception as e:  # noqa: BLE001
        if is_oom(e):
            result["oom"] = True
            result["error"] = "OOM: " + str(e)[:300]
        else:
            result["error"] = traceback.format_exc()[-3000:]
            print(result["error"], file=sys.stderr)
    finally:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=2)
    sys.exit(0 if (result["ok"] or result["oom"]) else 1)
