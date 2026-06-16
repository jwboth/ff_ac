# Klargjøring for Optuna-kalibrering (AC-serien)

Kort guide for å kjøre den grupperte Optuna-auto-kalibreringen lokalt.

## Forutsetninger

```powershell
cd ff_ac
git submodule update --init --recursive
uv python install 3.13
uv sync
.\.venv\Scripts\activate
```

## Kjør alt

```powershell
.\prepare_and_calibrate.ps1
```

Skriptet kjører i rekkefølge: protokoller → trykk → configer → grupper →
rigg-oppsett → seed fra AC60 → Optuna (10 representanter) → propager til medlemmer.

Nyttige flagg:
- `-SkipPrep` — hopp over data-prep (steg 1–2) hvis allerede gjort.
- `-RunAC60Calibration` — kjør AC60 color-embedding (interaktivt) hvis AC60 ikke
  allerede er kalibrert fra den tidligere ff_ac-testen.

## Hvorfor AC60 er utgangspunktet

Auto-kalibreringen er et **optimaliseringslag oppå en eksisterende kalibrering**:
`prepare_analysis_context(require_color_to_mass=True)` kaller
`HeterogeneousColorToMassAnalysis.load(...)`, altså den **laster** en kalibrering
og perturberer signalverdiene. AC60s kalibrering brukes derfor som **strukturmal +
startpunkt** for alle 10 representantene (samme rigg/fargestoff → fargestiene
overføres), og Optuna optimaliserer kun verdiene per representant (fanger
lysforskjellene mellom gruppene). `--baseline-trial true` evaluerer AC60-utgangs-
punktet uperturbert som trial 0 før søket starter.

## Det eneste manuelle steget

`color-embedding` (fargestiene per label) er interaktivt i nye DarSIA. Hvis AC60
allerede har dette fra den testede ff_ac-kjøringen, trengs ingen manuell innsats —
seedingen gjenbruker det for alle gruppene. Ellers: kjør med `-RunAC60Calibration`.

## Datagrunnlag — status (alle 43 forsøk validert)

- **42 av 43** forsøk får full injeksjonsprotokoll (27 intervaller).
- Robust håndtert: navnevarianter (`rest_5min`), splittet injeksjon (AC31),
  kort-reset (AC44), protokoll ett nivå opp (AC60/AC61), ark navngitt som forsøket
  (AC32 → `AC32.xlsx`), og **datofeil i mappenavn** (AC22: `221511` → måned 15;
  faller tilbake på EXIF).
- **AC38**: protokoll-arket mangler helt (reell datafeil). AC38 er ikke en
  gruppe­representant, så den blokkerer ikke Optuna-kalibreringen, men trenger et
  ark (eller en søsters protokoll som mal) før egen analyse.

## Trykk

Per forsøk: snitt-Lufttrykk fra Florida-stasjonen (hPa) fra injeksjonsstart til
+36 t, +40 m høydekorreksjon (×1.00463), konvertert hPa→bar. Stasjonsfilene må
dekke forsøksperioden (2022–2023). Temperatur er konstant 23 °C (artikkel s9).

## De 10 representantene

ac22, ac26, ac27, ac31, ac42, ac48, ac50, ac51, ac53, ac58
(se `config/calibration_groups/groups.json` for hele grupperingen).
