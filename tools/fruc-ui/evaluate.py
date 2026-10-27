"""Compare intermediate UI frames with their known synthetic positions.

Usage: evaluate.py FIXTURES BASELINE CANDIDATE
Each directory contains scroll.rgba and selection.rgba, except FIXTURES,
which contains scroll-reference.rgba and selection-reference.rgba.
Requires NumPy. Errors use RGB code values on the 0..255 scale.
"""
import json
import sys
from pathlib import Path

import numpy as np

fixtures, baseline, candidate = map(Path, sys.argv[1:])
regions = {
    "scroll": {
        "whole_image": (slice(None), slice(None)),
        "scrolling_text": (slice(365, 405), slice(None)),
        "fixed_header": (slice(0, 62), slice(None)),
    },
    "selection": {
        "whole_image": (slice(None), slice(None)),
        "menu_text": (slice(95, 440), slice(78, 360)),
        "moving_panel": (slice(112, 387), slice(650, 900)),
    },
}


def read(path):
    frames = np.fromfile(path, np.uint8).reshape(28, 540, 960, 4)
    # Every fifth output coincides with an input frame and is not synthesized.
    return frames[np.arange(28) % 5 != 0, :, :, :3].astype(np.float32)


results = {}
for scene, areas in regions.items():
    reference = read(fixtures / f"{scene}-reference.rgba")
    errors = [np.abs(read(root / f"{scene}.rgba") - reference)
              for root in (baseline, candidate)]
    results[scene] = {}
    for name, (ys, xs) in areas.items():
        old, new = [float(error[:, ys, xs].mean()) for error in errors]
        results[scene][name] = {
            "baseline_mae_8bit": old,
            "candidate_mae_8bit": new,
            "relative_reduction_percent": (old - new) / old * 100 if old else None,
        }
print(json.dumps(results, indent=2))
