# Testing status

Last updated: 9 November 2026. Tests below use synthetic fixtures unless explicitly described as release audits. There are no model-accuracy results yet.

| Check | Observed status | What it establishes |
|---|---|---|
| Windows Python core + CLI tests | **34 passed** | BIO/schema validation, leakage checks, exact entity metrics, overlap FP partition, subsets, replay probabilities, statistical pairing and bootstrap calculations |
| Ubuntu WSL2 core + CLI tests | **34 passed** | The same pipeline executes under local Linux, including conversion -> scoring -> document interval |
| Public release audit | **Completed** | Actual source counts, label/offset issues and duplicate inventory; see DATA_READINESS.md |
| TLUnified canonical export | **Completed** | Released splits preserved, valid BIO, no exact normalized cross-split duplicates |
| Nested target subsets | **Created** | Fixed 5/10/25% record subsets for seed 42; no model outcomes used |
| GitHub CI | Configured | Check the Actions tab for actual runs after push |
| WSL Parquet environment | **Created and tested** | Python 3.10 environment, PyArrow 20.0.0 installed, all 34 tests passed inside it |
| Notebook syntax | Checked | Code cells parse; this does not validate GPU execution |
| PyTorch prototype helper tests | Pending Kaggle environment | Tensor operations are outside the CPU standard-library suite |
| GPU smoke / tiny overfit / full baseline | **Not run** | Requires authenticated Kaggle launch and successful accelerator setup |
| Full proposed adaptation training | **Not implemented** | No prototype-drift or replay performance claims |

## Failure caught and corrected

The first WSL CLI integration run failed because `hashlib.file_digest` is absent in Python 3.10. Replaced it with streaming SHA-256, reran the affected suite, and obtained all 34 passes on both platforms. CI includes Python 3.10 to prevent recurrence.

Ubuntu initially lacked `python3.10-venv`; installing that standard package enabled the isolated data environment. Kaggle CLI 2.2.5 initially failed on an undeclared `six` import; `requirements-launch.txt` includes the verified fix. The CLI now starts, but the first authenticated request still reported that login was required.

## Reproduce core checks

```bash
python -m unittest discover -s tests -p 'test_core*.py' -v
```

The inherited PyTorch tests are deliberately separate and are invoked by the Kaggle notebook after dependencies are available. Do not interpret a green CPU workflow as proof of model training, CUDA compatibility, statistical power, or validity of unresolved corpus annotations.

## Upcoming decisions and gates

Use public TLUnified for the first GPU gate. Run one optimizer update, then overfit 32 training examples, then inspect a development-selected monolingual pilot. Preserve its code/data/model versions and logs. Freeze protocol choices and resolve remaining corpus adapters before final multi-language comparisons.

Before changing the experimental design or scaling compute, announce the decision and reason in the development chat. Report failures as well as passes and distinguish synthetic checks, corpus measurements and actual model results.
