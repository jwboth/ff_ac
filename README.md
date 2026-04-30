# ff_ac
Scripts for DarSIA analysis of the FF experiments in the AC series.

## Setup with uv (Python 3.13)

1. Clone the repository and initialize submodules:

   ```bash
   git clone https://github.com/jwboth/ff_ac.git
   cd ff_ac
   git submodule update --init --recursive
   ```

2. Install and use Python 3.13 with uv:

   ```bash
   uv python install 3.13
   uv sync
   ```

`darsia` is sourced from the local git submodule at `external/darsia` and installed by uv during `uv sync`.

3. Activate the virtual environment and get started:

```bash
.venv/Scripts/activate
python scripts/gui.py
```