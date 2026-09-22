"""Thin wrapper: QLoRA VRAM feasibility benchmark (GPU required).

Delegates to src.training.benchmark. Run from the repository root on a
CUDA machine (e.g. Google Colab T4):

    python scripts/colab_qlora_vram_benchmark.py
    python scripts/colab_qlora_vram_benchmark.py --seq-lengths 2048 4096 --batch-size 1
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/ is not a package; make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.benchmark import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
