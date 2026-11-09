# Philippine-language NER research foundation

Reproducible data preparation, exact entity scoring, and paired statistical analysis for a study of Tagalog, Cebuano, and Hiligaynon NER. The proposed method combines entity alignment with prototype-drift-guided source replay using XLM-R.

**Status:** CPU data/evaluation/statistics foundation implemented; Kaggle baseline runner prepared. GPU execution, full adaptation training, and research comparisons are not yet completed. No model-performance results are claimed.

## Follow the work

- [GitHub Actions](https://github.com/Ka1iZan/-Special-Problem-/actions): automated CPU checks after pushes.
- [Testing status](docs/TESTING_STATUS.md): what has actually run and what remains.
- [Workflow](docs/WORKFLOW.md): VS Code + WSL2 -> GitHub -> Kaggle.
- [Methodology review](docs/METHODOLOGY_REVIEW.md): scientific decisions and corrections to Chapters I-III.
- [Statistical analysis plan](docs/STATISTICAL_ANALYSIS_PLAN.md): estimands, paired design, multiplicity, and limits.
- [Data readiness](docs/DATA_READINESS.md): measurements from the actual public releases.

Major experimental decisions should be recorded before execution. Code checks establish correctness of the machinery; they do not demonstrate a useful research effect.

## Local CPU checks

Python 3.10+ is sufficient for the standard-library core:

```bash
python -m unittest discover -s tests -p 'test_core*.py' -v
```

Inside WSL2, run `bash scripts/setup_wsl.sh` to create `.venv-wsl`, install the Parquet reader, and run the checks. Select `.venv-wsl/bin/python` in VS Code. The local environment does not need CUDA or a model download. See the workflow for a fresh Linux-filesystem clone.

## Prepare the public pilot data

Activate the environment, then:

```bash
python scripts/fetch_public_data.py
python scripts/audit_public_releases.py --out artifacts/data_audit/public_releases.json
python scripts/prepare_tlunified.py
python scripts/sample_target.py data/processed/tlunified_v1/train.jsonl --seed 42 --out artifacts/subsets/tag_seed42.json
```

Downloads use fixed release revisions and record SHA-256 hashes. Output paths are versioned and refuse overwrite. TLUnified's released train/development/test split is preserved. The other corpora are downloaded for audit only; their format and ontology decisions remain open.

## Kaggle training gates

Import [`kaggle/spner_baseline.ipynb`](kaggle/spner_baseline.ipynb) into a **private** Kaggle notebook, enable internet and a GPU, and set `GIT_COMMIT` to the exact tested commit. It uses the public repository without a GitHub token.

Run `smoke` first (one optimizer update), then `tiny` (overfit 32 training records), then `baseline` after prior gates pass. Use a different `RUN_NAME` each time. The notebook does not evaluate the test split. It records code/model/data versions, dependencies, training logs and the selected checkpoint under `/kaggle/working/spner_runs/`.

## Analysis after predictions exist

```bash
python scripts/evaluate_predictions.py --gold data/processed/tlunified_v1/test.jsonl --predictions runs/selected/test.predictions.jsonl --out artifacts/selected.evaluation.json
python scripts/analyze_runs.py analysis/run_results.csv --out artifacts/analysis/principal
```

Copy `analysis/run_results.template.csv` to the ignored `analysis/run_results.csv` and populate it with observed, matched runs. The analysis enforces the declared comparison family and pairing metadata. `compare_predictions.py` additionally supports paired document or crossed run/document bootstrap intervals; genuine document IDs are required unless sentence resampling is explicitly selected.

## Data and confidentiality

This repository contains code, configuration, synthetic test fixtures, and shareable research notes. Raw/processed corpora, checkpoints, credentials, local reports, and private annotation material are excluded from Git. Dataset access and reuse remain subject to their source terms. Confidential FilNER data and documentation stay in local `private/`; they are not notebook inputs or repository assets.

The original `docs/TECHNICAL_DESIGN.md` remains local pending the author's review and is excluded from this initial commit. Use the methodology review above for current decisions. `configs/baseline.json` and `configs/analysis_plan.json` are the active pilot configurations. The older YAML configurations are design sketches, not a supported full-method runner.
