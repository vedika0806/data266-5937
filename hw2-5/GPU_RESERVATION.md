# GPU_RESERVATION.md — HW2.5

## Reservation Record
 - how did you access this GPU? 
 - Filled in the google form for request and got it approved.

## Actual Usage
- Session start: 2026-09-10 21:16:57 (from RUN_LOG.txt first timestamp)
- Session end: 2026-09-11 00:37:29 (from RUN_LOG.txt last timestamp)
- Approximate GPU-hours consumed: ~3.3 hours (derived from RUN_LOG.txt timestamps,
  since no explicit session-end timer was recorded in the notebook)
- GPU-hours reserved: 24 hours

## Note
No formal SESSION_START/SESSION_END timer was implemented in the notebook itself;
the usage window above is reconstructed from the first and last timestamps in
RUN_LOG.txt, which is traceable to the same GPU UUID (GPU-f0ce2985-2f23-4732-8ba8-cd27d18de22b)
throughout.
