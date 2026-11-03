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

llama.cpp's build or release commit is recorded here at M6 (P-12).
