# Reference hardware

Every benchmark and timing in this repo comes from this machine unless a table says otherwise. Recorded 2026-11-03 from WSL with `lscpu`, `free -b` and `lsb_release -a`.

| Item | Value |
|---|---|
| CPU | Intel Core i9-14900K: 24 cores (8 performance, 16 efficient), 32 threads. WSL's `lscpu` reports 16 cores per socket, 2 threads per core, 32 CPUs |
| RAM visible to WSL | 67,277,840,384 bytes (62 GiB) |
| GPU | None usable: no NVIDIA GPU, `nvidia-smi` absent. FinBERT and the local LLM run on CPU; vLLM and CUDA paths stay off |
| OS | Windows 11 Pro with WSL2, Ubuntu 22.04.5 LTS, kernel 6.18.54.1-microsoft-standard-WSL2 |
| Disk | WSL ext4 virtual disk on an internal 2 TB SATA HDD (`D:\WSL\Ubuntu\ext4.vhdx`); 853 GB free in `~` at M1 |
| Python | 3.12.15 in WSL and 3.12.10 on Windows, through `uv` 0.12.23 (WSL) and 0.12.21 (Windows) |

## Load on this machine

Live recorders share the PC: a crypto capture, a Deribit cron and Tardis downloads. Timings measured while they run are marked as such, and downloads are capped at 10 MB/s.

## Local inference runtime

llama.cpp `llama-server`, release `b11796` (2026-10-21, commit `0a1ef309cdbb9f2024f58a38fb4c42352c5db569`), Ubuntu x64 CPU build, installed by `scripts/serve_local_llm.sh install` and checked against its SHA-256 (`70b856d4...`). It serves `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (bartowski/Meta-Llama-3.1-8B-Instruct-GGUF at revision `bf5b95e9`, 4.92 GB, SHA-256 `7b064f58...`) on 127.0.0.1:8080 with 8 threads, an 8,192-token context, one slot and nice 10. The owner started it on 2026-11-04 (P-12).

Measured on 50 contamination-sample documents (2026-11-04, `m6-llama-bench50`): 50 of 50 valid answers, 32.2 s mean per document (p50 32.0 s, p95 38.1 s), 2,291 prompt tokens and 53 output tokens on average. Prompt evaluation ran at about 78 tokens per second and generation at about 4.6, with the system prompt served from llama.cpp's prompt cache after the first call. That is about 2,680 documents per day.
