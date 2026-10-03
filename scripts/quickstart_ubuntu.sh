#!/usr/bin/env bash
set -e

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[pdf,dev]'

if [ ! -f .env ]; then
  cp .env.example .env
fi
set -a
. ./.env
set +a

echo
echo "Python environment ready."
echo "Install Ollama if needed:"
echo "  curl -fsSL https://ollama.com/install.sh | sh"
echo "  ollama pull qwen3:8b"
echo
echo "Then run:"
echo "  esg-extract config"
echo "  esg-extract check"
echo "  pytest -q"
echo "  esg-extract extract data/examples/sample_esg_report.pdf --limit 5"
