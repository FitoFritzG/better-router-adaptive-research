# Dataset provenance and acquisition

## Selected source

The academic experiment uses the [**RouterBench 0-shot artifact**](https://huggingface.co/datasets/withmartian/routerbench/blob/a4dcf98b60f1faf85c572ee5f20cc0069ca0501a/routerbench_0shot.pkl) as its primary candidate dataset. RouterBench was introduced by Hu *et al.* as a benchmark for multi-LLM routing and reports more than 405,000 inference outcomes across 11 models. The public dataset card describes more than 30,000 prompts drawn from benchmarks including MBPP, GSM8K, Winogrande, HellaSwag, MMLU, and MT-Bench.

| Field | Pinned value |
|---|---|
| Paper | [*RouterBench: A Benchmark for Multi-LLM Routing System*](https://arxiv.org/abs/2403.12031) |
| arXiv | `2403.12031` |
| Dataset DOI | `10.57967/hf/1996` |
| Code repository | [`withmartian/routerbench`](https://github.com/withmartian/routerbench) |
| Dataset host | [Hugging Face `withmartian/routerbench`](https://huggingface.co/datasets/withmartian/routerbench) |
| Selected file | `routerbench_0shot.pkl` |
| Upstream revision | `a4dcf98b60f1faf85c572ee5f20cc0069ca0501a` |
| Expected bytes | `99,567,659` |
| Expected SHA-256 | `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d` |
| Provenance checked | `2026-07-13` |

The 0-shot file is selected because it provides a single controlled prompting condition and is substantially smaller than the raw 1.2 GB artifact. The study will declare its final subset explicitly and will not claim a complete reproduction unless every upstream outcome is used.

## License and redistribution decision

The RouterBench code repository declares the MIT License. The Hugging Face dataset card available on the provenance date does **not** declare a dataset license. Code licensing must not be assumed to cover the hosted data or the source benchmark content. Therefore:

- the upstream pickle file is downloaded locally and ignored by Git;
- it is not committed, mirrored, packaged, or redistributed by this project;
- users must review the upstream dataset card, terms, and source benchmark licenses before use;
- only an original synthetic fixture is committed for automated tests.

See [`../LICENSES.md`](../LICENSES.md) for the complete licensing matrix.

## Reproducible download

```bash
python -m better_router_adaptive.data.download \
  --destination data/raw/routerbench_0shot.pkl
```

The downloader pins the upstream revision and SHA-256. It streams to a temporary file in the destination directory, verifies the complete digest, then performs an atomic rename. It writes `data/raw/download_manifest.json` with the source, revision, retrieval time, byte count, checksum, and license status.

An existing file is reused only if its checksum matches. A mismatch fails closed unless `--force` is supplied; even then, the original remains untouched until a replacement download has passed verification.

## Security note: pickle

`routerbench_0shot.pkl` is a Python pickle. Loading an untrusted pickle can execute arbitrary code. The acquisition module **never deserializes it**. A later conversion stage must be separately reviewed, run in an isolated environment, and export a non-executable format before modeling.

## Repository data directories

- `data/raw/`: verified upstream downloads and their local manifest.
- `data/interim/`: isolated conversion outputs before canonical cleaning.
- `data/processed/`: canonical, validated experiment tables.

The contents of these directories are ignored except for `.gitkeep` files and small non-sensitive manifests explicitly approved for publication.
