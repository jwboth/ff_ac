# Handoff — ff_ac colour-correction calibration (open problem: ~2× over-detection)

> ## RESOLVED 2026-06-12 — the "~2× inflation" was an early-stop artifact, NOT colour correction
>
> Analysis of the two runs' tmp CSVs settled it:
>
> 1. **The colour-on fleet was stopped at ~245–320 trials/rep** (of 1650 budget) — barely past
>    the 150 random warmup. The uncorrected run completed ~1600–1850 trials/rep.
> 2. **At equal trial count the uncorrected run was just as bad**: truncating the uncorrected
>    CSVs to the colour run's row count gives best obj 0.00508 (= detect-nothing) for 8/10 reps.
>    The uncorrected optima only appeared later in its budget. The fleet comparison was
>    unconverged-vs-converged.
> 3. **Direct A/B at IDENTICAL params** (243 shared warmup rows per rep, same RNG seed):
>    colour-on detected mass is **0.85–1.0×** of uncorrected at every timepoint, all 10 reps.
>    Colour correction does not inflate the signal at all — it slightly reduces and smooths it.
>    The sRGB linearisation (§2 "prime suspect") is exonerated; no code change needed.
>    (ac51's last two points show on/off ≈ 1.24 — that's the correction fixing the measured
>    late-time lighting drop, i.e. working as intended.)
>
> **Action: re-run the colour-on fleet with the same command (§3) and let it finish the full
> budget.** Expect detect-nothing best rows until well past ~300–500 trials — the uncorrected
> run looked identical at that stage. §5's diagnostic plan is obsolete. The rest of this
> document remains valid as project context.
>
> ### Added 2026-06-12 (evening): physics of the residual over-detection + conservation penalty
>
> Even converged runs over-detect 1.3–1.6× late-time. Quantified root cause (water recipe:
> 9.07 g BTB + 1.00 g NaOH per 20 L → 1.25 mM alkalinity, 0.73 mM BTB): carbonate equilibrium
> (pK₁ 6.35, BTB pKa ≈ 7.1) gives colour-transition midpoint at dissolved CO₂ ≈ 1 mM and FULLY
> yellow at ≈ 3.9 mM, vs ~34 mM saturation. **The indicator saturates at ~11 % of CO₂
> solubility** — every pixel between 11 % and 100 % of saturation is optically identical, so
> the dilute fringe of the dissolved plume reads as saturated and detected mass tracks plume
> AREA (which grows after shut-in), not mass. That is why the ratio rises monotonically
> post-injection and why no static parameters can flatten it via the point-wise objective alone.
>
> Implemented mitigation (physics: closed cell after shut-in ⇒ true mass constant):
> **mass-conservation drift penalty** in `evaluate_run`
> (`scripts/auto_calibrate_color_to_mass.py`) — objective += λ · total-variation of detected
> mass over the post-injection plateau. Opt-in by reusing the already-plumbed flag:
> `--objective-integral drift` (λ=1) or `drift:0.5` etc.; the `choices=` restriction on the
> queue CLI was lifted. λ=1 is well-scaled: on the uncorrected ac53 best row the penalty is
> 0.00061 vs main term 0.00098. Metrics stay raw (penalty affects objective only), so
> detected/injected analyses are unchanged. NOTE: objectives are not comparable across runs
> with different λ.
>
> ### Added 2026-06-13: ff_um comparison + titration-anchored flash (IMPLEMENTED, opt-in)
>
> The uploaded ff_um run (`auto_calibration_run20.csv`) stays at ratio 0.94–1.04 to 96 h with
> the SAME indicator and SAME pipeline. It did NOT use any titration curve — plain linear
> `SimpleFlash` (min_value_aq 0, max_value_aq 1.0, min_value_g 1.0, max_value_g 1.5). The
> difference is the operating point: ff_um injected ~4.5e-3 vs AC ~4.7e-4 (~10× more CO₂), so
> the ff_um plume sits in the SATURATED colour band (yellow-fraction ~0.98, flat) where
> spreading doesn't change the colour integral and the linear ramp happens to work. AC's
> dilute plume sits in the STEEP transition band (72 % of the colour change is below 1.5 mM),
> so spreading moves more pixels into the responsive zone → detected mass rises. Same
> chemistry, different operating point — ff_um's success is consistent with, not a
> counterexample to, the titration analysis.
>
> **Implemented: `TitrationFlash`** in `external/darsia/.../multiphase/flash.py` — a
> `SimpleFlash` subclass whose AQUEOUS branch replaces the linear `c_aq(signal)` ramp with the
> inverse BTB/carbonate titration curve (derived here from the recipe; nothing titration-like
> existed in DarSIA). Gas branch unchanged. Signal convention: normalised position between
> `min_value_aq` (blue) and `max_value_aq` (full yellow) = normalised yellow fraction; only the
> SHAPE between them changes, so those params stay optimisable. Effect: a transition-midpoint
> pixel maps to c_aq≈0.03 (~1 mM) instead of 0.5 (~17 mM) — transition pixels stop being
> over-counted.
>
> Opt-in, NO rig rebuild (injected in `build_context`, applies in master + workers). TWO
> equivalent triggers:
>   - **STAMP (use this for the distributed/parallel queue)**: write
>     `config_seg6/run_ac/.titration_state/<run>.txt` = `on` (optionally `on 1.25,0.726,34` to
>     override the recipe = alk_mM,btb_mM,co2sat_mM). Mirrors the colour stamp: master + every
>     worker auto-follow it, so there is NO env threading and NO risk of mixed workers. Then run
>     the SAME parallel master command as §3 (+ watchdog/workers as usual) — nothing else changes.
>   - **ENV (handy for the one-process standalone)**: `$env:FFAC_TITRATION_FLASH="on"`
>     (`FFAC_TITRATION_RECIPE="1.25,0.726,34"`).
>
> Confirm it is live via the per-run log line `[<run>] TitrationFlash ACTIVE ...` (printed in
> master and each worker).
>
> **CAVEAT 1:** expect this to push AC toward UNDER-detection if the plume is genuinely dilute —
> that is the honest reading (the indicator can't see much mass there). **CAVEAT 2:** the
> optimiser may shrink the [min_value_aq, max_value_aq] window to push pixels toward full yellow;
> watch those params and consider tightening their bounds in a `bounds_seg6_titration.json`.
>
> Note: standalone (`auto_calibrate_color_to_mass.py --runs ac53`) runs all trials SERIALLY in
> one process (~100 s/eval × N trials) and has no `--bounds-file` (uses PARAM_SPACE_TEMPLATE) —
> fine for a quick look, but for a real budget use the parallel queue with the stamp.
>
> ac53 first test: write the stamp (or set env for standalone), colour-on, then compare
> `scripts/diag_ac53_ratios.py ac53` plateau ratios vs the linear-flash run.
>
> Further physics-anchored options discussed but NOT implemented: regime weighting (9/13
> points are post-injection plateau), coupling max_value_aq = min_value_g, shared node shape ×
> per-facies gain (~32 → ~11 dims), titration-derived signal→concentration curve from the
> recipe above.

Audience: a fresh chat with a newer model, continuing this project. Read all of this
before touching anything. The hard-won gotchas in §6 will save you hours.

---

## 0. TL;DR of the open problem (what to solve)

We added per-frame **colour correction** to the distributed auto-calibration of the Albus
(AC) FluidFlower CO₂ experiments. A single-rep standalone test looked great, but the **full
fleet run with colour correction over-detects CO₂ mass by ~2.7×** (detected/injected ≈ 2.7
across all timepoints), versus ~1.3–1.6× for the previous **uncorrected** fleet using the
**same bounds file**. So:

```
same --bounds-file, same reps:   uncorrected → 1.3–1.6×      colour-on → ~2.7×
```

**Colour correction roughly doubles the detected mass.** The flash parameters pin at their
upper bounds (`max_value_aq=1.0`, `max_value_g=1.75`) trying to reduce mass and still
over-detect, so the optimiser cannot compensate within the current bounds. For several reps
the optimiser's "best" is literally **detect-nothing** (all signal nodes 0.0) because every
non-zero trial over-detects so badly that zero is less wrong.

The fleet run was **stopped**. Your job: find WHY colour correction inflates the signal/mass
~2×, and decide whether it's fixable (bounds / linearisation / embedding) or whether colour
correction should be dropped.

Prime suspect: the **sRGB→linear linearisation** I added to the colour correction (it
stretches dynamic range → larger colour-path distances → higher signal → more mass). Second
suspect: the bounds file is simply too tight for the colour-shifted signal scale.

---

## 1. Project context

- Repo: `C:\Users\olav_\Documents\GitHub\ff_ac`. Vendored DarSIA at `external/darsia`. venv
  at `.venv` (`.\.venv\Scripts\python.exe`).
- Goal: distributed DarSIA auto-calibration (colour→CO₂-mass) for the AC FluidFlower series,
  ported from `ff_um`. 10 reps (ac22, ac26, ac27, ac31, ac42, ac48, ac50, ac51, ac53, ac58)
  + ac60, using **seg6** (6-segment per-facies colour paths → 7 nodes value0..value6,
  value0 locked at 0).
- The pipeline: colour → per-facies colour-path signal (pH proxy) → `SimpleFlash` (c_aq, s_g)
  → `mass = solubility_co2·c_aq·(1−s_g) + density_gaseous_co2·s_g`, integrated by
  `geometry.integrate` (auto-rescales voxel volume to data shape).
- Objective per rep: `sum over ~13 calibration timepoints | detected_total_mass − injected_mass |`.
  Injected CO₂ is small (~0.47 g total for ac53), gaseous (density 1.872 kg/m³), short
  injection (~2.5 h) then ~45 h rest.

### ABSOLUTE CONSTRAINT
Raw images in `C:\Users\olav_\Pictures\Albus` are **original data — READ ONLY, never modify**.
Work only on copies/results. (Results live on `Z:\Albus\Results\<rep>\`.)

---

## 2. What has been built / changed (state of the code)

All edits are in `C:\Users\olav_\Documents\GitHub\ff_ac`.

### scripts/auto_calibrate_color_to_mass.py (the per-run calibration engine)
- `build_context(...)`: builds the rig+analysis, preloads ~13 calibration images, computes
  injected mass per image. Called by both the standalone `main()` and the distributed queue.
- **Downscaling block** (`quality_scale != 1.0`): coarsens the rig to a target shape with
  per-step isolation (images, base/labels/mask, HeterogeneousModel.masks, CO₂ density/
  solubility, VolumeAveraging restoration mask). Prints `[QSCALE ...]` shapes. **NOTE: we
  decided to run full-res (`quality_scale=1.0`) so this block is bypassed now**, but it's
  correct if ever needed. (History: it took 5 hidden full-res arrays to make downscaling
  work — see §6.)
- **Neighbour substitution** (in the preload loop): if `read_image` flags a frame
  (ColorCorrection.last_flagged), substitute the nearest correctable neighbour within ±30 min
  (frames are 5 min apart), else drop the point. No-op when colour correction is off.
- **Colour-state stamp auto-follow**: reads `config_dir/.color_state/<run>.txt` (= `on`/`off`)
  and appends `config_seg6/coloron.toml` to the config list when `on`. This is the colour
  TOGGLE — no master flag, master+workers auto-follow the stamp the prep wrote.
- `PARAM_SPACE_TEMPLATE`: signal nodes bounded `(0, 2.0)`; flash currently **locked** to
  ff_um `(0, 0.75, 0.75, 1.0)`. **BUT the fleet uses `--bounds-file`, which OVERRIDES this
  template — so the lock had no effect in the fleet.**
- `[EVALTB ...]` one-shot traceback print in `evaluate_run`'s except (diagnostics).

### external/darsia/.../corrections/color/colorcorrection.py (the colour correction)
DarSIA's `ColorCorrection`. Active only when `[corrections.color]` is in the config. My edits
(all dormant unless colour correction is enabled):
- **#2 Linearisation** (`_srgb_to_linear`/`_linear_to_srgb`): in the `"darsia"` balancing
  branch, decode sRGB→linear before fitting/applying the affine balance, re-encode after.
  Controlled by config key `linearize` (default True). **PRIME SUSPECT for the ~2× inflation.**
- **#5/#6 Robust swatches**: replaced stochastic k-means in `CustomColorChecker._extract_from_image`
  with a deterministic per-channel **median** of the central 60 % of each swatch window.
- **#3 Residual self-check**: after correction, re-measure swatches, set `self.last_residual`
  + `self.last_flagged` (threshold `residual_warn_threshold`, default **0.06**), warn if over.
- `getattr(self, "active", False)` guard so a config-less ColorCorrection degrades to inactive
  instead of crashing.

### scripts/distributed_auto_calibration_queue.py
- Fixed an earlier crash: `_finalize_run` (module-level) called `_log_master` (a closure in
  `master_main`) → NameError → replaced with `print("[master] ...")`. (Master finalize
  re-evals best at full scale/float64 and writes `final_full_scale_<run>.json` +
  `auto_calibration_<run>.csv`.)
- Master clears the queue on startup by default (`--no-clear-queue` to skip).
- No colour flag was added (stamp design made it unnecessary).

### Config / toggle plumbing
- `config_seg6/coloron.toml`: overlay with `[corrections.color] colorchecker = "upper_right"`.
- `config_seg6/run_ac/ac53.toml`: `[corrections.color]` was REMOVED (base configs = colour-off).
- `config_seg6/run_ac/.color_state/<rep>.txt`: per-rep stamp (`on`/`off`) the prep writes.
- `config/bounds_seg6_coupled.json`: **the bounds the master actually uses**. Contents:
  `flash.max_value_aq [0.75,1.0]`, `flash.min_value_g [0.75,0.75]`, `flash.max_value_g
  [1.1,1.75]`, `signal.label*.value1..6 [0.0,2.0]`. (Despite "coupled" in the name, the data
  shows `max_value_aq ≠ min_value_g`, so coupling is NOT enforced.)

### Helper scripts (assistant-made; keep — they're the rollout tooling)
- `scripts/check_colorchecker.py <run>`: builds the rig for a rep (incl. coloron overlay),
  reports whether ColorCorrection is active + the ROI + the post-correction residual VERDICT
  (GOOD/BAD), and saves a shape-corrected upper-right crop. Used to verify checker detection.
- `prep_color_seg6.ps1 -Color {on|off} [-Runs ...]`: per rep runs `setup.py --rig` →
  `check_colorchecker` → `calibration.py --color-embedding` → `--default-mass --reset`, writes
  the stamp, redirects noisy output to `logs/prep/<rep>_<color>.log`, and writes an
  incremental summary CSV `logs/prep_color_<color>_<ts>.csv` (one concise terminal line/rep).
- `scripts/diag_ac53_ratios.py`: builds a rep's context and prints per-timepoint
  detected/injected ratios (the colour-state it follows comes from the stamp).

---

## 3. The two-step colour workflow (how it's meant to run)

1. **Prep (direct PowerShell, one-time per colour state)** — builds the cache (rig+embedding+
   seed) + writes stamps:
   ```
   .\prep_color_seg6.ps1 -Color on        # whole fleet WITH colour  (or -Color off)
   ```
2. **Fleet calibration (queue launcher / master CLI)** — auto-follows the stamps. The master
   command actually used:
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
   (Watchdog + workers started separately.) Resolution policy decided: **full-res +
   float32**, no spatial downscale (AC images are ~17 MP vs ff_um ~42 MP, so 0.5 was too
   aggressive and could erase the small free-gas signal). `float32` only affects the worker
   SEARCH; the master finalize re-computes best at full float64.

Colour correction is baked into the cache (rig `color_correction_*.npz` + the embedding/seed
built on corrected images). `load_corrections` loads the npz **regardless of config**, so the
colour state lives in the cache; toggling = re-prep (the user accepted this).

---

## 4. Key findings / evidence (don't re-derive these)

### 4a. The over-detection numbers (colour-on fleet, run `logs/...20260612_0014`)
Best-row detected/injected, full fleet, all `status=ok`, 0 eval-errors:
- ac53: ~2.6× early, **2.9× at 48 h**, best_obj 0.0050 (flash pinned: max_value_aq=1.0,
  max_value_g=1.75; signal nodes reach 2.0).
- ac22: ~2.4×, best_obj 0.0037.
- ac26 + 6 reps: **best = detect-nothing** (all signal nodes 0.0, obj 0.00508 = sum(injected)).

### 4b. Same colour correction worked standalone (so colour CAN work)
A standalone single-rep test (`scripts/auto_calibrate_color_to_mass.py --runs ac53 ...`,
full-res, 20 trials, **template param space, no bounds file**, colour-on) gave 48 h ratio
**1.56** (down from the uncorrected fleet's erratic 2.04), obj 0.00206. The fleet's only
difference from this is the **bounds file + warmup + trials** — the embedding is identical.
So the over-detection is a param-space/scale interaction, not a broken pipeline.

### 4c. The colour correction itself is high quality
Prep rolled out cleanly to all 11 reps: ColorCorrection active, residuals **0.0009–0.0038**
(threshold 0.06), all VERDICT GOOD. `find_colorchecker` reliably lands on the Classic
ColorChecker (the upper-right one; there are TWO charts in a Calibrite case — a creative chart
on the left, the Classic on the right — but detection/fallback picks the Classic correctly).
Residual flagging + neighbour substitution demonstrably fired once in production (ac51
DSC05072 at residual 0.061 → substituted DSC05071, 5 min away).

### 4d. Lighting vs physics (earlier analysis, still valid)
The late-time (40–48 h) **erratic** dip/spike in detected mass is **lighting instability**
(measured: the inert colourchecker's white patch jumps +10–23 % at 40–48 h while a CO₂-free
blue region stays blue → brightness, not hue, drift). Colour correction is meant to fix this.
The **smooth** mid-experiment over-detection (rising to ~1.6× by 48 h even uncorrected) is the
**BTB pH-indicator structural ceiling**: the indicator codes pH threshold, not concentration,
so a large dilute dissolved-CO₂ plume over-counts late and a fresh concentrated plume
under-counts early. Colour correction does NOT fix this (it's a method limitation).

### 4e. Flash semantics (so you reason correctly)
`min/max_value_aq` and `min/max_value_g` are **coordinates on the signal axis** (the pH-proxy,
range ~0–2 set by the signal-model node bounds), NOT saturations. Outputs c_aq, s_g are always
normalised to [0,1]. So `max_value_g=1.75` just means "gas saturates at signal 1.75". Raising
`max_value_aq`/`max_value_g` WIDENS the ramps → LESS mass per signal. The optimiser raising
both to their ceilings = it's trying to cut the over-detection and running out of room.
Physical coupling (water saturation = gas onset) would set `max_value_aq = min_value_g`; it is
NOT enforced in the current run. Whether flash should be static vs floating: static is more
physical IF the signal scale is consistent across reps — but colour correction shifts the
scale, so static flash mismatches and the flash NEEDS to float to absorb it (this is why the
"lock flash to ff_um" idea was wrong here).

### 4f. Overflow scale data (if it comes up)
`FF Albus/scale_probe/AC50_weights.csv` is the overflow weight (water pumped in to replace
dissolving-gas volume). Its growth RATE is the informative signal, but it's dominated by a
constant ~370 g/h pumping background that stops ~13 h, so the small gas-dissolution signal is
buried; not a clean free-gas measure without the pump-rate log.

---

## 5. Diagnostic plan for the open problem (recommended order)

Work on ONE rep (ac53) for fast iteration; the user runs the commands (you can't run DarSIA —
see §6).

1. **Isolate linearisation.** Set `linearize=False` for the colour correction (either default
   in `colorcorrection.py` `_init_from_config`, or pass via the rig's ColorCorrection config),
   re-prep ac53 `-Color on`, re-run `scripts/diag_ac53_ratios.py ac53` (it follows the stamp).
   If the ~2× drops markedly → the sRGB→linear stretch is the inflation source → either remove
   it or re-tune. This is the highest-probability single cause.
2. **Controlled colour A/B at matched settings.** ac53 colour-on vs colour-off, SAME bounds
   file, SAME trial budget (use the standalone `auto_calibrate_color_to_mass.py` with
   `--bounds-file config/bounds_seg6_coupled.json` for a fair fleet-equivalent). Quantify how
   much colour alone inflates detected mass with these exact bounds.
3. **Widen the flash bounds.** Make a `bounds_seg6_colour.json` with e.g. `max_value_aq
   [0.75, 1.5]`, `max_value_g [1.1, 3.0]`. Re-run ac53. If the optimiser now reaches ratio ≈ 1
   → it was purely bounds (the colour-shifted signal needs wider flash to normalise) → fine,
   re-tune bounds for colour. If it still over-detects → the inflation is upstream (colour
   correction / embedding), not the flash.
4. **Compare signal magnitudes.** Extract best-row `signal.label*.value*` for ac53 colour-on
   vs colour-off. If colour-on signal nodes are systematically higher/pinned at 2.0 → colour
   inflates the colour-path distance (consistent with linearisation). The relative_colorpath
   is `corrected_image − corrected_baseline`; the baseline is also corrected, so a consistent
   correction should NOT inflate the difference — if it does, the correction is non-uniform
   (linearisation, or per-frame detection variance).
5. **Decide.** If colour is salvageable with `linearize=False` and/or wider flash bounds →
   re-prep + re-run fleet. If not → run the fleet `-Color off` (the validated 1.3–1.6× path)
   and treat colour as future work. The colour mechanics (toggle, prep, flagging,
   substitution) are all built and verified; only the signal-scale interaction is unresolved.

Quick win for tonight if needed: `.\prep_color_seg6.ps1 -Color off` then run the fleet — gives
usable uncorrected calibrations (1.3–1.6×, same as before).

---

## 6. Environment & gotchas (CRITICAL — these wasted a lot of time)

- **You (the assistant) cannot run DarSIA.** The sandbox lacks `plotly` (so `import darsia`
  fails) and has no access to the `Z:` results drive or `\\Moderskipet` network paths. The
  USER runs every `setup.py` / `calibration.py` / queue / prep command and pastes output. You
  read repo files and the `logs/` CSVs (the distributed-queue tmp CSVs DO contain per-timepoint
  `metrics`, unlike the standalone CSV which is only `iter,objective`).
- **Mount staleness.** When editing large files, the assistant's bash mount serves
  **truncated/stale** copies (e.g. a 740-line file appears cut at ~680 lines mid-statement),
  so `ast.parse` via bash gives false SyntaxErrors. The **Read tool (Windows) is
  authoritative**. To verify a Python edit: Read the region, and/or `cp` to `/tmp` and parse an
  ISOLATED snippet of the new code. Don't trust a bash full-file parse of a just-edited large
  file.
- **PowerShell**: the assistant can't run or parse `.ps1`; the user runs it.
- **Calibration path resolution** (so you don't rediscover it): `results` is per-rep in
  `run_ac/<rep>.toml` (e.g. `Z:\Albus\Results\ac53`). Rig cache = `<results>/setup/rig`
  (built by `setup_rig` → `rig.setup(...)` + `rig.save(...)`, triggered by
  `scripts/setup.py --rig`). Colour paths = `<results>/calibration/color/color_paths/...`,
  seed = `.../color_to_mass/...`. `prepare_analysis_context` always `cls.load(config.rig.path,
  ...)` — it LOADS the cached rig, never rebuilds; so changing `[corrections.color]` requires
  re-running `setup.py --rig`. `--color-embedding` only LOADS the rig (does not rebuild it).
- **DarSIA path merge**: `_get_section_from_toml(path_list, ...)` deep-merges configs in order,
  so `coloron.toml` appended last adds `[corrections.color]` on top of the run's corrections.
- **The bounds file beats the param-space template.** `--bounds-file` overrides
  `PARAM_SPACE_TEMPLATE`. Edit the JSON, not the template, to change fleet bounds.
- **Two colour charts** in the image (Calibrite case): left = creative/spectral, right =
  Classic 24-patch (the one DarSIA uses, mounted portrait). Detection handles it; don't
  "fix" it.
- Use the task list tool, ask clarifying questions before big multi-file changes, and prefer
  ONE-rep tests before fleet-wide actions.

---

## 7. Quick reference

| Thing | Where |
|---|---|
| Repo | `C:\Users\olav_\Documents\GitHub\ff_ac` |
| Raw images (READ ONLY) | `C:\Users\olav_\Pictures\Albus` |
| Per-rep results / caches | `Z:\Albus\Results\<rep>\` (setup/rig, calibration/color, ...) |
| Config | `config_seg6/common.toml`, `config_seg6/run_ac/<rep>.toml` |
| Colour overlay | `config_seg6/coloron.toml` |
| Colour stamps | `config_seg6/run_ac/.color_state/<rep>.txt` |
| Fleet bounds (authoritative) | `config/bounds_seg6_coupled.json` |
| Colour correction code | `external/darsia/src/darsia/corrections/color/colorcorrection.py` |
| Calibration engine | `scripts/auto_calibrate_color_to_mass.py` |
| Queue | `scripts/distributed_auto_calibration_queue.py` |
| Prep orchestration | `prep_color_seg6.ps1` |
| Checker verify | `scripts/check_colorchecker.py <run>` |
| Ratio diagnostic | `scripts/diag_ac53_ratios.py` |
| Latest (broken) colour run | `logs/facies1_perlabel1_warmup150_optuna1500_parallel_20260612_0014/` |
| Prior uncorrected run | `logs/facies1_perlabel1_warmup150_optuna1500_parallel_20260610_0050/` |

Reps: ac22 ac26 ac27 ac31 ac42 ac48 ac50 ac51 ac53 ac58 (+ ac60 prepped but not in fleet).

**First action for the new chat:** confirm the open problem by reading the two runs' tmp CSVs
(compare best-row detected/injected per timepoint, colour-on vs uncorrected), then start the
§5 diagnosis with the linearisation test on ac53.
