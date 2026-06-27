#!/usr/bin/env bash
# GovRAG Copilot — launch the Gradio demo UI
# Builds the index if needed, then starts the bilingual web app.
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY="$(which python)"
fi

# Ensure the index exists
if [[ ! -f "data/index/hybrid.pkl" ]]; then
    echo "→ Index not found, building it now ..."
    "$PY" src/build.py
    echo
fi

echo "🛡️  Starting GovRAG Copilot UI on http://localhost:7860"
echo "   (Ctrl+C to stop)"
echo
echo "   Backend selection (env var GOVRAG_BACKEND):"
echo "     • extractive  — pure-Python, no LLM (default fallback)"
echo "     • ollama      — local Ollama server at :11434"
echo "     • hf          — HuggingFace transformers"
echo

exec "$PY" ui/app.py
