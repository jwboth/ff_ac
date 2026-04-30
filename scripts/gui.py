"""CLI for DarSIA workflows GUI."""

import logging

from darsia.presets.workflows.user_interface_gui import launch_workflows_gui

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    launch_workflows_gui()
