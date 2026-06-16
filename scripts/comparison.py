"""CLI for cross-run comparison (events / Wasserstein).

Mirrors ff_ac's analysis.py / calibration.py entrypoints, exposing the new
DarSIA ``preset_comparison``. Driven by the multi-run config produced by
``generate_wasserstein_config.py``.

    python scripts/comparison.py --config config/wasserstein_ac.toml --wasserstein-compute
    python scripts/comparison.py --config config/wasserstein_ac.toml --wasserstein-assemble
"""

import logging

from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.user_interface_comparison import preset_comparison

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    preset_comparison(Rig)
