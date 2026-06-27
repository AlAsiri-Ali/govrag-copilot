"""
GovRAG Copilot - Build script
==============================
End-to-end pipeline: ingest PDFs -> chunk -> build hybrid index.

Run from project root:
    python -m src.build
or
    cd src && python build.py
"""
import sys
from pathlib import Path

# Make sibling modules importable when running this file directly
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ingest import ingest_directory       # noqa: E402
from index import HybridRetriever, build_index  # noqa: E402


def main() -> None:
    root = HERE.parent
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    index = root / "data" / "index"

    print(f"[1/2] Ingesting documents from {raw} ...")
    ingest_directory(raw, processed)

    print(f"\n[2/2] Building hybrid retrieval index ...")
    build_index(processed, index)

    print("\n✅ Build complete.")
    print(f"   Chunks:  {processed/'chunks.jsonl'}")
    print(f"   Index:   {index/'hybrid.pkl'}")


if __name__ == "__main__":
    main()
