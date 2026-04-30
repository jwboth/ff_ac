"""CLI for utils."""

import logging

from darsia.presets.workflows.user_interface_utils import preset_utils

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    preset_utils()
