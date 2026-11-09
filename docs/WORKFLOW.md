# VS Code + WSL2 -> GitHub -> Kaggle

| Place | Work | Evidence to retain |
|---|---|---|
| Local WSL2 + VS Code | Data audits, adapters, protocol, CPU tests, Git commits | Test output, data hashes, reviewed changes |
| GitHub | Shareable code/configuration and CI | Commit SHA, workflow results |
| Private Kaggle notebook | GPU smoke test, tiny overfit, then training | Notebook version, dependencies, data/model/code hashes, logs and checkpoints |

Repository: `https://github.com/Ka1iZan/-Special-Problem-.git`. It is public. The development branch is `codex/research-foundation`. Keep raw data and restricted material outside the tracked tree. The initial notebook uses public TLUnified data only.

## Local setup

Ubuntu under WSL2 is installed. The current workspace on the Windows drive is readable from WSL. For sustained preparation, a separate clone in the Linux filesystem avoids cross-filesystem and OneDrive overhead; see [VS Code's WSL guide](https://code.visualstudio.com/docs/remote/wsl). The existing workspace need not be moved.

In an Ubuntu terminal, after the first push:

```bash
mkdir -p ~/projects
cd ~/projects
git clone --branch codex/research-foundation https://github.com/Ka1iZan/-Special-Problem-.git spner
cd spner
bash scripts/setup_wsl.sh
code .
```

Use the WSL extension and select `.venv-wsl/bin/python`. If Ubuntu lacks `venv`, install the matching `python3-venv` package before rerunning setup. CPU tools require Python 3.10+. Parquet dependencies are pinned separately for Python 3.10 and 3.11+; no local CUDA installation is required.

The `.env.example` paths are optional storage conventions from the starter kit. Current scripts take explicit paths or repository-relative defaults; they do not automatically load `.env`.

## Development loop and live visibility

1. Make a small code/protocol change and record its scientific purpose.
2. Run `python -m unittest discover -s tests -p 'test_core*.py' -v`.
3. Review `git diff` and `git status --short`, including newly untracked files.
4. Commit reviewed code and push to the development branch.
5. Inspect [GitHub Actions](https://github.com/Ka1iZan/-Special-Problem-/actions). Each push runs CPU checks on Python 3.10, 3.11 and 3.13.
6. Use the exact successful commit SHA in Kaggle.

Working files in VS Code and progress messages in the development chat are the live local view. GitHub shows committed/pushed snapshots. Kaggle cell output becomes the GPU run log once a run is actually started.

## Kaggle launch

Import `kaggle/spner_baseline.ipynb` into a private notebook. Enable internet and select an available GPU. Set `GIT_COMMIT` to the full SHA from the tested GitHub commit. `PRIVATE_REPO=False` is correct; no GitHub token is needed.

Run all cells. The notebook installs its requirements while retaining Kaggle's PyTorch, runs CPU and prototype helper tests, downloads pinned public data, prepares TLUnified, and executes the selected gate. Package compatibility and model download remain part of GPU validation.

| Gate | Setting | Pass condition |
|---|---|---|
| Smoke | `MODE="smoke"` | One optimizer update, finite loss/logits and nonzero gradient; provenance saved |
| Tiny overfit | `MODE="tiny"`, new run name | At least 0.95 exact entity F1 on 32 training records within the diagnostic limit |
| Baseline pilot | `MODE="baseline"`, new run name | Full training completes with development-selected checkpoint and matching saved predictions |

The tiny gate uses training examples for its diagnostic score; it is not a generalization estimate. Failure returns a nonzero exit and needs investigation before scaling up. Overlength examples fail preflight and require explicit windowing.

Save a Kaggle version with outputs to retain `/kaggle/working/spner_runs/<run_name>`. Record the notebook/version URL. The pilot runner cannot resume; use unique names and preserve completed artifacts.

The notebook does not pass `--evaluate-test`. Test evaluation is appropriate after model-selection rules are frozen; then the runner reloads the best checkpoint and stores test hashes and predictions.

## CLI authentication and monitoring

The researcher selected CLI launch and monitoring. The current [Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md) requires Python 3.11+. Windows Python 3.13 can host this launch client while CPU development stays in WSL2 Python 3.10. A newer isolated WSL Python is also suitable; do not replace Ubuntu's system Python.

For the project Windows environment, in PowerShell at the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-launch.txt
.\.venv\Scripts\kaggle.exe auth login
.\.venv\Scripts\kaggle.exe kernels list --mine --page-size 1
```

Complete the browser authentication yourself. An alternative is the official token file at the user's `.kaggle/access_token`, or the legacy `.kaggle/kaggle.json`; keep credentials out of chat and Git. The agent checks availability/authenticated commands without printing token contents.

Create a separate ignored launch directory containing only the pinned notebook and `kernel-metadata.json` adapted from `kaggle/kernel-metadata.example.json`. Use a unique private kernel slug for each diagnostic. Then use the documented [kernel commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md):

```text
kaggle kernels push -p <launch-directory>
kaggle kernels status <username/kernel-slug>
kaggle kernels output <username/kernel-slug> -p <local-output-directory>
```

Authentication, accelerator eligibility, quota and successful execution must be observed; creating a notebook does not establish any of them. Do not publish the notebook or attach restricted data to get a diagnostic running.

## Run provenance and scope

Each run records configuration, selected record IDs, data hashes, resolved model revision, Git revision/dirty state, Python/Torch/CUDA/package versions, dependency freeze, training logs, best checkpoint, selected predictions and completion status. Device-level determinism is reported rather than promised.

This is an ordinary monolingual XLM-R pilot. Complete multi-source adaptation, prototype monitoring and replay training loops remain future implementation. Statistical scripts need actual paired run records; an empty results template is not evidence.
