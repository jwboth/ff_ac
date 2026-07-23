# Final AC calibration campaign (2026-07-23)

## Method

- 40 experiments: all configured AC runs except AC29 and AC51.
- AC29 and AC51 are excluded as physical/chemical initial-state outliers.
- Full per-label signal model (not the reduced shared-shape model).
- Physically derived `TitrationFlash`.
- Per-image Colorchecker correction, except the documented AC44 color-off cache.
- Strict partial-affine registration to the AC14 segmentation template.
- Equal-weight point-wise L1 objective over up to 16 calibration frames.
- The established 13 frames plus 4.1, 6, and 8 h as explicit early
  redistribution constraints.
- A requested frame is used only when raw data exist within its configured
  tolerance; a late endpoint is never reused for a later nominal time.
- `config/bounds_seg6_titration.json`.
- 59 trusted prior full-model winners evaluated first.
- 243 structured warmups, 150 seeded random warmups, then 1600 Optuna trials.
- Stable Optuna seed 17.
- Saved signal and flash models are persistent production outputs.

The nominal campaign contains 79,778 worker evaluations, plus occasional sanity
evaluations. One master keeps at most 24 runs active and permits exactly one
task per run, filling 24 workers across two machines without concurrent Optuna
suggestions for the same experiment.

## Preparation

```powershell
cd C:\Users\olav_\Documents\GitHub\ff_ac
.\.venv\Scripts\python.exe scripts\ac_final_calibration_prepare.py prepare
```

Preparation writes the seed file, a preflight report, launch scripts, and a
backup of every existing signal/flash calibration under:

`Z:\Albus\Autokalibrering_log\final_production_20260723_16frames_24x1`

## Launch

Primary machine (one master and 12 workers):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'Z:\Albus\Autokalibrering_log\final_production_20260723_16frames_24x1\start_master_and_12_workers.ps1'
```

Second machine (12 workers):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'Z:\Albus\Autokalibrering_log\final_production_20260723_16frames_24x1\start_12_workers.ps1'
```

## Stop

Run this on each machine. It terminates the full local campaign process tree and
fails if any campaign worker remains:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'Z:\Albus\Autokalibrering_log\final_production_20260723_16frames_24x1\stop_campaign_on_this_machine.ps1'
```

## Held-out validation

After all 40 final JSON files exist, evaluate raw frames near 3.5, 7, and 12 h.
These frames exist as distinct images in all 40 experiments and are rejected if
they overlap any training frame.

```powershell
cd C:\Users\olav_\Documents\GitHub\ff_ac
.\.venv\Scripts\python.exe scripts\ac_final_calibration_validate.py validate
```
