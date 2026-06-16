# ff_ac distribuerte arbeidsflyter (port fra ff_um)

Disse skriptene gjenskaper Word-dokumentets fire kommandogrupper for AC-serien,
bygget på den nye DarSIA i `external/darsia`. Det API-uavhengige kø-laget ligger
i `queue_core.py`; selve fagvitenskapen (analyse/kalibrering/Wasserstein) kjøres
via ff_ac sine egne preset-CLI-er (`analysis.py`, `calibration.py`,
`comparison.py`), som workerne kaller.

## Nye filer

| Fil | Rolle |
|-----|-------|
| `queue_core.py` | Delt kø: atomisk claim, heartbeat, retry, stale-requeue, control-dir worker-grenser, tasks-per-file. Watchdog starter worker-*prosesser*. |
| `_workflow_common.py` | Felles argparse + worker-kommando-kjøring (stdout-parsing). |
| `generate_protocols.py` | `imaging_protocol_*.csv` + `injection_protocol.csv` per forsøk. **Ren protokoll-basert tid – ingen EXIF/bildelesing**: bildetid = intervall-starttid (mappenavn = protokollens «Start interval imaging»-tid) + i·intervall; injeksjonsanker = t=0-stegets «target»-tid på forsøksdatoen. Samme referanseklokke for begge → rutenettet er justert. Robust mappegjenkjenning: navnevarianter (`rest_5min`), splittede faser (AC31), kort-reset (AC44, ordinal spacing), protokoll ett nivå over (AC60/61), dato gjenfunnet fra protokoll-arket ved mappenavn-tastefeil (AC22 221511→221115), og `*_overflow.xlsx` hoppes over til fordel for `Albus protocol NN.xlsx` (AC18/19/20). |
| `generate_configs.py` | `acNN.toml` per forsøk + felles tidsrutenett (102 punkter, begge injeksjoner). |
| `generate_pressure.py` | `pressure_temperature_protocol.csv` per forsøk fra Florida-stasjonen (hPa): snitt start..+36t, +40m høydekorreksjon, hPa→bar. Konstant fallback. |
| `generate_wasserstein_config.py` | Multi-run `wasserstein_ac.toml` for den nye `comparison_wasserstein`. |
| `comparison.py` | Entrypoint for `preset_comparison` (events / Wasserstein). |
| `distributed_analysis_queue.py` | Masseanalyse: `master`/`watchdog`/`worker`/`merge`. |
| `distributed_auto_calibration_queue.py` | Autokalibrering: `master`/`watchdog`/`worker`/`best`. |
| `distributed_wasserstein_queue.py` | Wasserstein: `supervisor`/`watchdog`/`worker`/`assemble` (finkornet kø) + `compute`/`assemble-darsia` (ekte DarSIA) + budsjettsjekk. |
| `calibration_objective.py` | Masse-balanse-objektiv mot DarSIA: setter signalverdier → `cta(img).mass` → `geometry.integrate` → vs `injected_mass(date)`. |
| `calibration_worker.py` | Per-trial worker som kjører objektivet og skriver verdien. **Koblet til DarSIA.** |
| `group_calibration.py` | Clustrer 43 forsøk → ~10 kalibreringsgrupper (farge/lys i første bilder). |
| `apply_calibration_groups.py` | Seeder representanter fra AC60 (`--seed-from`), og pusher optimaliserte representanter til medlemmer. |
| `wasserstein_worker.py` | Per-oppgave finkornet W1-worker. **Stub** – DarSIA-kallet gjenstår (den sekvensielle `compute`-løypa virker uten). |

## Distribuert / server-bilder

For batch-oppgaver fordelt på flere maskiner leser workerne bildene fra serveren.
`[data].folders` + `[protocols.imaging]` i hver `acNN.toml` inneholder de absolutte
bildestiene — sett `--data-root "Z:\\Albus"` ved config-generering (gjort).
Protokoll-CSV-ene inneholder kun *filnavn* og er portable; generer dem lokalt
(raskt) på én maskin, så deler alle maskinene repoet (configer + `protocols/`) og
leser bildene fra `Z:\\Albus`. Køen (`--queue`) bør ligge på en delt sti alle når.

## Forutsetning

```powershell
cd ff_ac
git submodule update --init --recursive
uv python install 3.13
uv sync
.venv\Scripts\activate
```

## Pipeline

```powershell
# 1) Protokoller for alle forsøk (ren protokoll-tid, ingen EXIF + injeksjonsrampe)
python scripts/generate_protocols.py --albus-root "Z:\Albus\Raw data" --out protocols --all
python scripts/generate_pressure.py   --protocols-root protocols --all `
  --pressure-xlsx "..\Florida*.xlsx" --altitude-diff-m 40 --window-hours 36   # snitt start..+36t, +40m korr, hPa->bar

# 2) Config per forsøk + Wasserstein-config
python scripts/generate_configs.py --albus-root "C:\Users\olav_\Pictures\Albus" `   # --albus-root: LOCAL (scan/EXIF)
  --out config/run_ac --data-root "Z:\Albus\Raw data" --results-root "Z:\Albus\Results" --all       # --data-root: SERVER (workers read here)
python scripts/generate_wasserstein_config.py --out config/wasserstein_ac.toml --results-root "Z:\Albus\Results" --resize 0.10

# 3) Masseanalyse  (worker kjører: analysis.py --config common.toml acNN.toml --mass --all)
python scripts/distributed_analysis_queue.py master   --queue \\share\Darsia_Queue --runs ac17 ac18 ... --analysis mass --all
python scripts/distributed_analysis_queue.py watchdog --queue \\share\Darsia_Queue --workers 14 --control-dir \\share\Darsia_Queue\worker_limits --stop-when-drained
python scripts/distributed_analysis_queue.py merge     --queue \\share\Darsia_Queue --allow-missing

# 4) Autokalibrering (grupper) - AC60 seeder strukturmal + startpunkt
python scripts/group_calibration.py --albus-root "C:\Users\olav_\Pictures\Albus" --out config/calibration_groups --groups 10
python scripts/apply_calibration_groups.py --groups-file config/calibration_groups/groups.json --results-root "E:\ff_ml4gcs" --seed-from ac60
python scripts/distributed_auto_calibration_queue.py master   --queue \\share\Kalibrering --groups-file config/calibration_groups/groups.json `
  --baseline-trial true --max-iters 500 --param-ranges "value1=0,2;...;value6=0,2" --param-levels "value1=8;...;value6=8"
python scripts/distributed_auto_calibration_queue.py watchdog --queue \\share\Kalibrering --workers 12 --control-dir \\share\Kalibrering\worker_limits --stop-when-drained
python scripts/distributed_auto_calibration_queue.py best      --queue \\share\Kalibrering
# etter Optuna: push hver optimaliserte representant til sine gruppemedlemmer
python scripts/apply_calibration_groups.py --groups-file config/calibration_groups/groups.json --results-root "E:\ff_ml4gcs"

# 5) Wasserstein – budsjettsjekk mot ff_um, så ekte DarSIA-compute
python scripts/distributed_wasserstein_queue.py supervisor --queue Z:\FF_AC\Analysis\Wasserstein --runs 17 18 ... 61 --points-per-run 102 --rois full box1 box2 --dry-run
python scripts/distributed_wasserstein_queue.py compute          --config config/wasserstein_ac.toml
python scripts/distributed_wasserstein_queue.py assemble-darsia  --config config/wasserstein_ac.toml
```

`supervisor --dry-run`: alle 43 forsøk × 102 punkter × 3 ROI = 276 318 distanser
= **90 % av ff_um-taket** (108 137 par / 306 726 ROI-distanser). Rutenettet er tett
gjennom begge injeksjonene (I1 0–0,9 t, I2 1–2,4 t) og eksponentielt glissere etterpå.

## Validering mot AC60 (kjent-fungerende baseline)

`config/run_ac/ac60.toml` (generert) matcher den testede `config/run_jakub/ac60.toml`
på alle kritiske felt – samme baseline (DSC19883), run_id, results, korreksjoner og
labeling. Forskjeller er bevisste tillegg: hvilefase, felles tidsrutenett, EXIF-tider.
**Anbefalt første smoke-test etter `uv sync`:** kjør analyse på AC60 og sammenlign mot
det kjente resultatet:
```powershell
python scripts/analysis.py --config config/common.toml config/run_ac/ac60.toml --mass --all
```

## Verifisert i sandkasse (uten DarSIA)

Kø-kjernen (enqueue/claim/complete, retry→failed, control-dir, stale-requeue);
protokoll-generator (AC53: 373+541 EXIF-bilder, 27 injeksjons­intervaller, begge porter; robust mappegjenkjenning verifisert på AC17/AC31/AC44 – navnevariant, splittet injeksjon, kort-reset);
config-generator (alle 43, gyldig TOML, 102 rutenett-punkter, begge injeksjoner); Wasserstein-multi-config
(gyldig, 43 runs, korrekt skjema); kalibrerings-objektivets parameter-logikk (parse/apply, global + per-label,
monoton clipping) enhetstestet med mock; alle CLI-er ende-til-ende (master→watchdog worker-pool→merge/assemble/best);
korrekte preset-kommandostrenger.

## Gjenstår å koble på (TODO i koden)

- **Autokalibrering** (`calibration_objective.py`): objektivet er nå skrevet mot DarSIA-API-et
  (`prepare_analysis_context` → sett `signal_model.model[1][label]`-verdier → masse vs injisert masse).
  Parameter-logikken er enhetstestet; selve evalueringen må kjøres/valideres etter `uv sync`. Mulige
  forbedringer: Optuna i stedet for tilfeldig søk i master, per-label-parametre, og caching.
- **`wasserstein_worker.py`** (finkornet kø-løype): kall `darsia.wasserstein_distance` på to
  oppløste massefelt. Den ekte sekvensielle løypa (`compute`/`assemble-darsia`) virker uten dette.
- **Finkornet parallell Wasserstein**: den nye `comparison_wasserstein` eksponerer ikke
  `skip_existing` på CLI – legg til det flagget i `user_interface_comparison.py` for ekte
  fler-worker-parallellisering, ev. bruk kø-løypa.
- **Fysiske konstanter (bekreftet mot artikkelen):** `NOMINAL_SCCM = 10 ml/min` er MFC-ens **fullskala**
  (s3), brukt som %-nevner for «Flow %». **Faktisk injeksjonsrate maks 2.0 ml/min** (ramp 0.1→1.5→2.0, s9).
  Density 1.872 (std CO2 20 C/1013 mbar, s4). To porter I1 (x=0.45) / I2 (x=0.72) fra ac14-protokollen.
- **Trykk/temperat