#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 SA-AT scaling benchmark 上评估 HRLF 基线。"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sa_at.evaluate_learning_on_sa_at_dataset import main


if __name__ == "__main__":
    main(default_method="hrlf")
