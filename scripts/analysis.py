"""CLI for analysis."""

import logging

from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.user_interface_analysis import preset_analysis

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    preset_analysis(Rig)
