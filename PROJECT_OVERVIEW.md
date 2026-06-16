# ff_ac — Project overview (for a fresh start, e.g. Codex)

Thorough description of the whole project: what it is, the grouping strategy, the calibration
pipeline, the scientific challenge, everything tested, current state, and the gotchas. A
companion doc `HANDOFF_color_calibration.md` has the deep detail on the colour-correction and
titration-flash work; this file is the wider map.

---

## 1. What the project is

We process the **Albus (AC) FluidFlower** CO₂ experiments. Each experiment is a thin
sand-filled cell saturated with water + a **BTB pH indicator** (blue at high pH → yellow when
CO₂ acidifies it). Gaseous CO₂ is injected at the bottom for ~2.5 h, then the cell rests ~45 h.
A camera shoots the cell every few minutes. **Goal: turn the colour images into a CO₂-mass map
and a detected-total-mass-vs-time curve**, then compare detected mass to the known injected
mass (from the mass-flow controller / injection protocol).

Stack: Python, **DarSIA** (vendored at `external/darsia`, with our edits), Optuna for the
calibration optimisation, a custom file-based distributed queue. Repo:
`C:\Users\olav_\Documents\GitHub\ff_ac`. venv `.\.venv\Scripts\python.exe`. This is a port/
evolution of an older `ff_um` project (same imaging method, different rig/experiments).

### The colour→mass pipeline (per image)
```
image ─corrections→ colour ─per-facies colour path→ signal (pH proxy, ~0..2)
      ─SimpleFlash→ (c_aq dissolved, s_g gas) ─→ mass = solubility·c_aq·(1−s_g) + density·s_g
      ─geometry.integrate→ detected total mass
```
- **Corrections** (applied at image read): type(float32), drift (geometric alignment using the
  colour-checker as a landmark), curvature (crop/warp), and optionally **colour correction**.
- **Colour paths / "seg6"**: each *facies* (sand type) has a colour-path curve through colour
  space from blue (no CO₂) to yellow (saturated), with **6 segments → 7 nodes** value0..value6;
  value0 locked at 0. The signal is the position along that path (a pH proxy).
- **Facies anchoring**: several layers are the *same* sand (same physics) but look slightly
  different due to position/lighting. With `--use-facies`, same-sand layers share one
  calibration; all facies are anchored to a **reference facies (F, label 5)** which defines the
  common signal scale, and value0=0 is the common zero. See HANDOFF §4e / the colour doc for
  the flash semantics.
- **Flash**: maps signal → (dissolved c_aq, gas s_g). Parameters are signal-axis coordinates
  (NOT saturations); outputs are normalised [0,1]. `mass = solubility_co2·c_aq·(1−s_g) +
  density_gaseous_co2·s_g`, the density/solubility maps come from depth/pressure
  (`setup_density`).
- **Objective** (per experiment): `Σ over ~13 calibration timepoints | detected − injected |`.
  Optuna optimises the per-facies signal nodes (+ optionally flash params) to minimise it.

---

## 2. The grouping strategy (calibrate one experiment per group, propagate to the rest)

There are **43 AC experiments**; calibrating all is too expensive. So:

1. **Cluster by colour/lighting** (`scripts/group_calibration.py`): for each experiment read
   the first few *pre-injection* frames, extract a feature vector (per-channel mean+std +
   brightness on a downscaled image), z-score across experiments, **k-means into ~10 groups**.
2. **Representative** = the experiment closest to its cluster centroid.
3. **Calibrate only the ~10 representatives** (these are the reps we work with: **ac22, ac26,
   ac27, ac31, ac42, ac48, ac50, ac51, ac53, ac58**; `ac60` is a template/seed). Outputs
   `config/calibration_groups/groups.{csv,json}` = `{group: {representative, members}}`.
4. **Propagate** (`scripts/apply_calibration_groups.py`): after Optuna, push each
   representative's calibration to its group MEMBERS, so **all 43 experiments reuse a
   calibration**. (Also used up front to seed all representatives from ac60 as a template +
   Optuna start point.)

So everything in the distributed calibration below operates on the **10 representatives**, and
the group structure fans the result out to all 43. Example group: representative AC26 covers
{AC23, AC25, AC26, AC28, …}.

---

## 3. The distributed calibration infrastructure

`scripts/distributed_auto_calibration_queue.py` — a file-based master/watchdog/worker queue:
- **master**: builds per-rep contexts, creates Optuna studies, enqueues warmup + optuna tasks,
  reads worker results, writes per-rep tmp CSVs (`logs/<run_tag>/tmp_auto_calibration_<run>.csv`,
  with per-timepoint `metrics`), and at the end **finalises** (re-evaluates best at full-res/
  float64, writes `auto_calibration_<run>.csv` + `final_full_scale_<run>.json`). Clears the
  queue on startup by default.
- **watchdog** spawns N **workers**; workers claim tasks, call `build_context` + `evaluate_run`
  in `scripts/auto_calibrate_color_to_mass.py`, and write results back.
- **Per-rep search**: 150 random **warmup** trials + up to 1500 **optuna** trials.
- **Bounds**: `--bounds-file config/bounds_seg6_coupled.json` defines the param ranges and
  **OVERRIDES** the `PARAM_SPACE_TEMPLATE` in `auto_calibrate_color_to_mass.py`. (Editing fleet
  bounds = edit the JSON, not the template.)
- **Resolution policy (decided)**: **full resolution + float32** (`--quality-scale 1.0
  --quality-dtype float32`). AC images are ~17 MP (vs ff_um ~42 MP), so the 0.5 spatial
  downscale we first used was too aggressive (and risked erasing the small free-gas signal).
  float32 only affects worker SEARCH evals; the master finalise re-computes best at full
  float64, so the saved result is full precision.
- `queue_launcher.py` is a tkinter GUI that auto-discovers the CLI args and builds/launches the
  master/watchdog commands (the prep below is run directly, not via the launcher).

The actual master command in use:
```
.\.venv\Scripts\python.exe scripts\distributed_auto_calibration_queue.py master ^
  --queue "C:\Users\olav_\Documents\Darsia_Queue\Kalibrering" ^
  --runs ac31 ac26 ac42 ac22 ac27 ac48 ac51 ac50 ac53 ac58 ^
  --config-dir config_seg6/run_ac --use-facies true --per-label true ^
  --bounds-file config/bounds_seg6_coupled.json ^
  --max-iters 1500 --warmup-iters 150 --run-mode parallel --max-in-flight-per-run 3 ^
  --control-dir "\\Moderskipet\Darsia_Queue\Kalibrering\control" ^
  --sanity-every 100 --sanity-scale 1.00 --quality-dtype float32
```

---

## 4. THE core scientific challenge (the thing that actually limits accuracy)

**The BTB pH indicator saturates far below CO₂ saturation, so detected mass tracks plume AREA,
not mass, and over-detects the dilute dissolved plume late in the experiment.**

Quantified from the water recipe (9.07 g BTB + 1.00 g NaOH per 20 L → ~1.25 mM alkalinity,
~0.73 mM BTB; carbonate pK₁ 6.35, BTB pKa ≈ 7.1): the colour-transition midpoint is at
dissolved CO₂ ≈ **1 mM**, fully yellow at ≈ **3.9 mM**, vs ~**34 mM** saturation. So the
indicator is optically saturated at ~**11 % of CO₂ solubility** — every pixel between 11 % and
100 % of saturation looks identical. After shut-in the plume *spreads* (convective dissolution),
so its dilute fringe reads as fully saturated and **detected mass rises monotonically even
though true mass is constant** (closed cell). This is a *method* limitation, not a bug, and no
static colour→mass parameters can flatten it through the point-wise objective alone. Typical
converged over-detection: **1.3–1.6× by 48 h** (uncorrected), rising smoothly.

(There is ALSO a separate, smaller effect: late-time **lighting instability** — the inert
colour-checker's white patch jumps +10–23 % at 40–48 h while a CO₂-free blue region stays blue
→ brightness drift, not hue. This causes *erratic* late dips/spikes; colour correction targets
this. Distinguish the two: smooth rise = BTB ceiling/physics; erratic jumps = lighting.)

ff_um (same indicator, same pipeline) stays at ratio 0.94–1.04 to 96 h — NOT because it's
better, but because it injected ~10× more CO₂, so its plume sits in the *saturated* colour band
(flat, spreading doesn't change the colour integral). AC's dilute plume sits in the *steep*
transition band, where spreading moves pixels into the responsive zone. Same chemistry,
different operating point.

---

## 5. What we built / changed (and why)

### 5a. Downscaling (quality_scale) — built, then retired in favour of full-res
Making worker evals run at 0.5 scale required coarsening 5 hidden full-res arrays (images,
analysis base/labels/mask, `HeterogeneousModel.masks`, CO₂ density/solubility, and the
`VolumeAveraging` restoration mask). It works (per-step isolated, prints `[QSCALE ...]`), but we
decided **full-res + float32** instead (§3). The block is bypassed at scale 1.0.

### 5b. Colour / lighting correction — built and rolled out (SCARE RESOLVED)
DarSIA `ColorCorrection` (auto-detects the upper-right Classic ColorChecker via
`find_colorchecker`). Our edits in `external/darsia/.../colorcorrection.py`: sRGB→linear
linearisation before the affine balance (#2); deterministic median swatch sampling (#5/#6);
post-correction residual self-check + `last_flagged` (threshold 0.06, #3). Plus **neighbour
substitution** in `build_context` (flagged calibration frame → nearest correctable neighbour
±30 min, else drop) and a **colour TOGGLE** via a per-rep stamp
(`config_seg6/run_ac/.color_state/<run>.txt` = on/off; `build_context` auto-appends
`config_seg6/coloron.toml` when "on"; master+workers auto-follow — no flag threading). Colour
state is baked into the cache (rig `color_correction_*.npz` + embedding + seed), so toggling =
re-prep. Orchestrated by `prep_color_seg6.ps1 -Color {on|off}` (runs `setup.py --rig` →
`check_colorchecker` → `--color-embedding` → `--default-mass --reset`, writes the stamp + a
summary CSV). **Rollout to all 11 reps was clean** (residuals 0.0009–0.0038, all GOOD).

**The "colour correction over-detects ~2×" alarm was a FALSE ALARM** (see the RESOLVED banner at
the top of `HANDOFF_color_calibration.md`): the colour-on fleet had been stopped at ~245–320
trials/rep (barely past the 150 warmup), and the uncorrected run looked identically bad at the
same trial count — it was *unconverged-vs-converged*. A direct A/B at identical params shows
colour-on detected mass is **0.85–1.0×** of uncorrected (it slightly *reduces* and smooths,
exactly as intended; the linearisation is exonerated). **Action: just let the colour-on fleet
run to the full budget.**

### 5c. Flash: static vs floating vs titration-anchored
- The flash params are signal-axis coordinates. Locking them to ff_um values is *more physical*
  only if the signal scale is consistent — it isn't perfectly, so the bounds file lets the flash
  float a little (`max_value_aq [0.75,1.0]`, `min_value_g [0.75]`, `max_value_g [1.1,1.75]`,
  signal nodes `[0,2]`).
- **`TitrationFlash`** (`external/darsia/.../multiphase/flash.py`, opt-in): a `SimpleFlash`
  subclass whose aqueous branch replaces the linear `c_aq(signal)` ramp with the **inverse
  BTB/carbonate titration curve** derived from the recipe. It makes a transition-midpoint pixel
  map to c_aq ≈ 0.03 (~1 mM) instead of 0.5 (~17 mM), so dilute transition pixels stop being
  over-counted. Opt-in via a stamp `config_seg6/run_ac/.titration_state/<run>.txt = on` (mirrors
  the colour stamp; master+workers auto-follow) or env `FFAC_TITRATION_FLASH=on`. Confirm via the
  `[<run>] TitrationFlash ACTIVE` log line. CAVEAT: expect it to push toward UNDER-detection if
  the plume is genuinely dilute (the honest reading — the indicator can't see that mass).

### 5d. Mass-conservation drift penalty
Physics: closed cell after shut-in ⇒ true mass constant ⇒ a good calibration should give a flat
detected-mass plateau. So `evaluate_run` can add `objective += λ · total-variation(detected mass
over the post-injection plateau)`. Opt-in via `--objective-integral drift` (λ=1) or `drift:0.5`.
λ=1 is well-scaled (on uncorrected ac53: penalty 0.00061 vs main term 0.00098). Metrics stay raw
(penalty affects objective only). NOTE: objectives aren't comparable across different λ.

---

## 6. Current state & open items

- **Colour-on fleet**: re-run with the §3 command and **let it finish the full 1650-trial
  budget** (expect detect-nothing best rows until ~300–500 trials — that's normal early-stop
  behaviour, the uncorrected run was identical there). The 2× scare is resolved.
- **Titration flash**: first real test pending — write the titration stamp for ac53, run, and
  compare `scripts/diag_ac53_ratios.py ac53` plateau ratios vs the linear-flash run. Expect the
  late-time over-detection to drop (possibly to under-detection).
- **Drift penalty**: available (`--objective-integral drift`) but not yet adopted fleet-wide.
- **Propagation**: once the 10 representatives are calibrated, run
  `apply_calibration_groups.py` to fan out to all 43.
- Ideas discussed, NOT implemented: regime weighting (9/13 timepoints are the post-injection
  plateau), enforce coupling `max_value_aq = min_value_g`, shared node-shape × per-facies gain
  (~32 → ~11 dims), a titration-derived signal→concentration curve.

---

## 7. Environment, constraints, gotchas (these cost real time)

- **ORIGINAL IMAGES ARE READ-ONLY**: `C:\Users\olav_\Pictures\Albus` — never modify. Work on
  copies/results only (results on `Z:\Albus\Results\<rep>\`).
- An **assistant cannot run DarSIA** in its sandbox (no `plotly`; no `Z:` / `\\Moderskipet`
  access). The USER runs every `setup.py` / `calibration.py` / queue / `.ps1` command and pastes
  output. The assistant reads repo files and `logs/*.csv` (the distributed tmp CSVs carry
  per-timepoint `metrics`; the standalone CSV is only `iter,objective`).
- **Mount/file staleness**: when editing large files, bash served truncated copies → false
  `ast.parse` SyntaxErrors. The Read tool (Windows) is authoritative; verify edits by Reading the
  region or parsing an isolated `/tmp` snippet of the new code.
- **Calibration is CACHED.** `prepare_analysis_context` always LOADS the cached rig
  (`<results>/setup/rig`), never rebuilds. So enabling colour/changing corrections requires
  re-running `scripts/setup.py --rig`; `--color-embedding` only loads the rig. `load_corrections`
  loads `color_correction_*.npz` if present **regardless of config**, so colour state lives in the
  cache → toggling = re-prep. `--bounds-file` overrides the template. DarSIA deep-merges the
  config list, so an overlay appended last (coloron.toml) adds its sections on top.
- Two colour charts in frame (Calibrite case): left creative, right Classic 24-patch (the one
  DarSIA uses, mounted portrait) — detection handles it.
- Prefer ONE-rep tests before fleet-wide actions; use the task list; ask before big multi-file
  changes.

---

## 8. Quick reference

| Thing | Path |
|---|---|
| Repo / venv | `C:\Users\olav_\Documents\GitHub\ff_ac` / `.\.venv\Scripts\python.exe` |
| Raw images (READ ONLY) | `C:\Users\olav_\Pictures\Albus` |
| Results / caches (per rep) | `Z:\Albus\Results\<rep>\` |
| Grouping | `scripts/group_calibration.py`, `scripts/apply_calibration_groups.py`, `config/calibration_groups/groups.json` |
| Calibration engine | `scripts/auto_calibrate_color_to_mass.py` (build_context, evaluate_run, PARAM_SPACE_TEMPLATE) |
| Distributed queue | `scripts/distributed_auto_calibration_queue.py` |
| Launcher GUI | `queue_launcher.py` |
| Config | `config_seg6/common.toml`, `config_seg6/run_ac/<rep>.toml` |
| Fleet bounds (authoritative) | `config/bounds_seg6_coupled.json` |
| Colour overlay / stamps | `config_seg6/coloron.toml`, `config_seg6/run_ac/.color_state/<rep>.txt` |
| Titration stamps | `config_seg6/run_ac/.titration_state/<rep>.txt` |
| Colour correction code | `external/darsia/.../corrections/color/colorcorrection.py` |
| Flash / TitrationFlash | `external/darsia/.../multiphase/flash.py` |
| Prep orchestration | `prep_color_seg6.ps1` |
| Checker verify / ratio diag | `scripts/check_colorchecker.py <run>`, `scripts/diag_ac53_ratios.py <run>` |
| Detailed colour/titration handoff | `HANDOFF_color_calibration.md` |
| Workflow notes | `scripts/README_workflows.md` |

10 representatives: ac22 ac26 ac27 ac31 ac42 ac48 ac50 ac51 ac53 ac58 (+ ac60 template). Group
structure fans these out to all 43 experiments.
