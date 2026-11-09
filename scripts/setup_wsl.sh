#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required for local data tools"'
python3 -m venv .venv-wsl
.venv-wsl/bin/python -m pip install -r requirements-data.txt
.venv-wsl/bin/python -m unittest discover -s tests -p 'test_core*.py' -v
printf 'WSL data environment ready. Select .venv-wsl/bin/python in VS Code. GPU training runs on Kaggle.\n'
