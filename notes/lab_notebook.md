# Lab notebook

Dated decisions, surprises, failed attempts. Newest at the bottom.

## 2026-11-01
- Plan revision 2 approved (see `plan.md`, amendments A1-A8 at the top win over the body).
- Decision: **CPU only.** Amendment A2 asked for CPU-vs-GPU timing. The machine has no CUDA
  (AMD Ryzen 9 5900HS, integrated Radeon graphics) and torch 2.15.0 is the CPU build.
  Pilot timings are CPU only, and every result JSON records `"device": "cpu"`.
