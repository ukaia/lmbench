#!/bin/sh
# Run lmbench, creating the venv on first use.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
    echo "First run: creating venv and installing textual + httpx …"
    python3 -m venv .venv || exit 1
    .venv/bin/pip install --quiet "textual>=1.0" "httpx>=0.27" || exit 1
fi
exec .venv/bin/python lmbench.py
