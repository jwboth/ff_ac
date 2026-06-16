# Undersøkelse: AC60-oppsett, fargestier, og automatisk fargesti-deteksjon

Ren kartlegging – ingenting er endret eller implementert.

## 1. Hvordan AC60 settes opp slik ff_ac krever

Tre steg (etter `uv sync` + aktivert venv), alle drevet av `common.toml` + `ac60.toml`:

```powershell
# a) Rigg-oppsett (én gang; deles av alle AC-forsøk)
python scripts/setup.py --config config/common.toml config/run_ac/ac60.toml --all
#   bygger: dybdekart, labels (fra data/DSC28507_segmented.png), facies,
#   protokoll-maler, rigg + korreksjonspipeline (curvature/drift/color).

# b) Color-embedding (fargestier per facies-label)
python scripts/calibration.py --config config/common.toml config/run_ac/ac60.toml --color-embedding

# c) Color-to-mass (modellen auto-kalibreringen laster og perturberer)
python scripts/calibration.py --config config/common.toml config/run_ac/ac60.toml --default-mass
```

`--mass` gir en manuell/finjustert massekalibrering; `--default-mass` en standard.
Auto-kalibreringens objektiv laster denne (`HeterogeneousColorToMassAnalysis.load`).

## 2. Er fargestier allerede lagret i ff_ac?

**Nei.** Søk i repoet (utenom `external/`) fant **ingen** lagrede artefakter
(`color_paths`, `signal_model`, `metadata.json`, `flash`, color-range). Repoet
inneholder kun *inputene*: `data/DSC28507_segmented.png` (segmentering/labels),
`data/facies.csv`, `data/depth_measurements.csv`, og config. Fargestiene **må
beregnes** av color-embedding-steget (2b). De lagres til `embedding.color_paths_folder`
under forsøkets results-mappe (`E:\ff_ml4gcs\ac60\...`).

Kalibreringsbildene som brukes (`common.toml`):
`calibration1` = 10 min–2,5 t (5 bilder), `calibration2` = 3–48 t (5 bilder) etter
start – dvs. bilder der CO₂ er til stede i varierende konsentrasjon.

## 3. Manuell vs. automatisk – og det viktigste funnet

`common.toml` har i dag `calibration_mode = "manual"` under
`[color.path.relative_colorpath]`. Men i DarSIA-kilden
(`signals/color/color_path_regression.py`, `find_color_path`):

```python
mode: Literal["auto", "manual"] = "auto"   # default!
# "auto"   -> returnerer det automatiserte resultatet
# "manual" -> starter fra auto-resultatet og åpner interaktiv key-color-editering
```

Det betyr at **den automatiske fargesti-deteksjonen finnes allerede** i DarSIA.
`manual` legger kun et *valgfritt* interaktivt redigeringssteg
(`_manual_postprocess_color_path`, en matplotlib-figur) oppå auto-resultatet. Alle
andre plott er bak `if verbose:` (styrt av `--show`).

**Konsekvens:** Setter man `calibration_mode = "auto"` (og lar `--show` være av),
blir color-embedding **helt ikke-interaktiv**. Da kan hele kjeden – inkludert
AC60-utgangspunktet – kjøres uten et eneste manuelt klikk.

### Hvordan auto-algoritmen virker (kort)

Per facies-label, fra kalibreringsbildenes fargespekter:
1. 1D-embedding: lineær regresjon som mapper farge → relativ konsentrasjon.
2. Origo-deteksjon: baseline-fargen ( from baseline-bilder, med støy-spekter ignorert).
3. Stykkevis-lineær sti (`num_segments = 3`) gjennom de «signifikante» fargeboksene,
   vektet med histogram (WLS). Resultatet er `key_colors` = fargestien.

## 4. Initielle tanker om «automatisk fargesti-deteksjon» (ikke implementert)

Siden DarSIA allerede har en auto-modus, handler «automatisering» mer om
**robusthet og kvalitetssikring** enn om å bygge fra bunnen:

- **Prøv `auto` først på AC60.** Hvis auto-stien er god nok, forsvinner hele det
  manuelle steget, og «seed fra AC60 + Optuna» optimaliserer videre derfra.
- **Kvalitetsmål / gate:** automatisk flagg når auto-stien er tvilsom (f.eks.
  ikke-monoton 1D-embedding, høyt regresjonsresidual, eller for få signifikante
  bokser i en label) → kun da kreves manuell finjustering.
- **Lukk loopen mot masse:** color-to-mass-objektivet (masse vs. injisert masse,
  som auto-kalibreringen alt bruker) kan i prinsippet brukes til å *velge mellom*
  auto-stier eller justere `num_segments`/vekting – en datadrevet seleksjon i stedet
  for visuell.
- **Gruppe-nivå:** fargestier trenger neppe egen deteksjon per forsøk – én auto-sti
  per kalibreringsgruppe (de 10) holder, på linje med dagens grupperingsplan.
- **AC60 som prior:** bruk AC60s (kjent-gode) sti som utgangspunkt og la auto kun
  justere innenfor en begrenset avstand – reduserer risikoen for at auto sklir ut.

**Anbefalt neste eksperiment (lavt risikonivå):** kjør AC60 color-embedding med
`calibration_mode = "auto"` og inspiser de lagrede `..._01_embedding.png` /
`..._02_origin_detection.png` (auto-modus lagrer disse når `directory` er satt) for
å vurdere kvaliteten før man bestemmer om manuell finjustering trengs i det hele tatt.

---

## 5. Bekreftelser fra DarSIA-oppsettsdokumentet (ff_glass_beads-supplement)

Det offisielle oppsettsdokumentet bekrefter og presiserer funnene over. Merk at
dokumentet bruker eldre script-navn (`--color-path`, `--color-signal`,
`--mass`); ff_ac (nyere) bruker `--color-embedding`, `--mass`, `--default-mass`.

**Offisiell rekkefølge:** Setup (data → protokoller → metadata → dybde → labels →
rigg) → Kalibrering (2.1 fargespektra/-stier → 2.2 kontinuerlige signaler →
2.3 masse) → Analyse. Steg 1–2 må kjøres i rekkefølge før analyse.

**Fargestier er automatiske (bekreftet, s10–11):** «DarSIA now features
**automatic detection of color spectra** for given calibration images. These
color spectra will then be used to build reliable color paths with a locally
linear structure.» Den eneste reelle brukerinputen er å **velge representative
kalibrerings- og baseline-bilder** (vi har allerede definert disse som
`calibration1/2` + `baseline` i `common.toml`).

**Det manuelle masse-steget = nøyaktig det Optuna automatiserer (s15):** masse-
kalibreringen åpner et interaktivt vindu med slidere der målet er å «**overlay the
injected and total mass**» (juster signal-transformasjonen til detektert totalmasse
overlapper injisert masse). Dette er presis samme objektiv som
`calibration_objective.py` (Σ|detektert − injisert masse|). **Optuna-auto-
kalibreringen erstatter altså den manuelle slider-tuningen.** `--default-mass` gir
en ikke-interaktiv start-massekalibrering som Optuna deretter optimaliserer.

**Det gjenstående potensielt-interaktive steget** er color-**signal**-finjusteringen
(s13–14: tegn rektangel + juster start/midt/slutt-verdier per facies). I nyere
ff_ac kan dette være foldet inn i `--color-embedding`; verifiser om det har et
ikke-interaktivt standardvalg på din DarSIA-versjon.

**Andre nyttige bekreftelser:**
- s5: «DarSIA implicitly interprets the **earliest time of reported activity in the
  injection protocol as official start of the experiment**» → bekrefter
  rutenett-ankeret (`experiment_start = injection_protocol["start"].min()`).
- s5: injeksjonsrate i **SCCM + density ved standardbetingelser** → samme format
  som vår `injection_protocol.csv` (`rate_sccm`, `density_kg/m3`).
- s5: trykk-temperatur «single line → constant condition» → vår konstant/snitt-
  tilnærming er gyldig (varierende betingelser også støttet).
- s4: imaging-protokoll = `path` + `datetime`; «image_id is a relict and is
  ignored» → samsvarer med vår CSV (image_id-kolonnen er uvesentlig).
- s8: manuell segmentering med dominante farger → `data/DSC28507_segmented.png`
  er denne (allerede levert i ff_ac).

**Synteser:** Hele kalibreringskjeden for AC60 er i praksis: setup (auto, men
krever den manuelle segmenteringen som alt finnes) → color-embedding (auto fargesti-
deteksjon) → `--default-mass` (ikke-interaktiv start) → **Optuna** (automatiserer
masse-overlappingen). Det eneste som kan kreve et klikk er color-signal-finjustering;
ellers er kjeden klikkfri.
