# Licensing and attribution matrix

| Component | Source | License status | Repository policy |
|---|---|---|---|
| Better Router Adaptive source code | This repository | MIT | Redistributable under `LICENSE` |
| RouterBench evaluation code | [`withmartian/routerbench`](https://github.com/withmartian/routerbench) | MIT declared by upstream repository | Cite upstream; no vendored code currently included |
| RouterBench hosted dataset | [Hugging Face `withmartian/routerbench`](https://huggingface.co/datasets/withmartian/routerbench) | **No dataset license declared on the dataset card as checked 2026-07-13** | Download locally only; do not commit, mirror, or redistribute |
| Benchmarks represented inside RouterBench | MBPP, GSM8K, Winogrande, HellaSwag, MMLU, MT-Bench, and others | Varies by source benchmark | Review source licenses before any redistribution or publication of row-level content |
| `tests/fixtures/routerbench_sample.csv` | Original synthetic fixture created for this project | MIT as part of this repository | May be redistributed; must never be described as experimental evidence |

## Required citations

RouterBench paper:

> Q. J. Hu, J. Bieker, X. Li, N. Jiang, B. Keigwin, G. Ranganath,
> K. Keutzer, and S. K. Upadhyay, “RouterBench: A Benchmark for Multi-LLM
> Routing System,” arXiv:2403.12031, 2024.

Dataset DOI: `10.57967/hf/1996`.

## Interpretation

An MIT license in the RouterBench GitHub code repository governs that code. It
does not automatically grant rights over separately hosted data, model
responses, or benchmark questions. Until the dataset owner publishes explicit
terms, this project takes the conservative position of non-redistribution.
