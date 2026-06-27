#!/usr/bin/env bash
# GovRAG Copilot — one-shot setup
# Installs Python dependencies and builds the retrieval index.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "🛡️  GovRAG Copilot — setup"
echo "=========================="
echo

# 1. Detect Python
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "❌ python3 not found. Please install Python 3.10+ first."
    exit 1
fi
PY_VERSION=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✓ Found Python $PY_VERSION"

# 2. Optionally create a venv
if [[ "${SKIP_VENV:-0}" != "1" ]] && [[ ! -d ".venv" ]]; then
    echo
    echo "→ Creating virtual environment in .venv ..."
    "$PY" -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY="$(which python)"
    echo "✓ Virtualenv ready"
else
    if [[ -d ".venv" ]]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
        PY="$(which python)"
        echo "✓ Reusing existing .venv"
    fi
fi

# 3. Install requirements
echo
echo "→ Installing Python dependencies ..."
"$PY" -m pip install --upgrade pip --quiet
"$PY" -m pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# 4. Verify documents are present
echo
echo "→ Checking source documents in data/raw ..."
required_docs=(
    "PersonalDataProtectionLawEn.pdf"
    "PersonalDataProtectionLawAr.pdf"
    "ImplementingRegulationPersonalDataProtectionLawEn.pdf"
    "ImplementingRegulationPersonalDataProtectionLawAr.pdf"
    "RegulationonPersonalDataEn.pdf"
    "RegulationonPersonalDataAr.pdf"
)
missing=0
for f in "${required_docs[@]}"; do
    if [[ -f "data/raw/$f" ]]; then
        echo "   ✓ $f"
    else
        echo "   ✗ MISSING: data/raw/$f"
        missing=1
    fi
done
if [[ "$missing" == "1" ]]; then
    echo
    echo "❌ Some source documents are missing. Place them in data/raw/ and re-run."
    exit 1
fi

# 5. Build the index
echo
echo "→ Building retrieval index ..."
"$PY" src/build.py
echo

# 6. Quick smoke test
echo "→ Smoke test: asking a sample question ..."
"$PY" - <<'PY'
import sys; sys.path.insert(0, "src")
from pipeline import GovRAGPipeline
p = GovRAGPipeline()
ans = p.answer("Within how many hours must a personal data breach be notified?")
print("  Q: Within how many hours must a personal data breach be notified?")
print(f"  A (preview): {ans.answer.splitlines()[1][:120]}...")
print(f"  Citations: {len(ans.citations)} ({ans.citations[0]['label']}, ...)")
PY

echo
echo "✅ Setup complete!"
echo
echo "Next steps:"
echo "  • Launch the demo UI:        ./scripts/run_demo.sh"
echo "  • Run the test suite:        pytest tests/ -v"
echo "  • Run the eval suite:        python src/evaluate.py"
echo "  • Use a local LLM (Ollama):  see README.md → 'LLM Backends'"
