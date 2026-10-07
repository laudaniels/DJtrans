# DJtransGAN vanavond draaien op de RTX (8GB)

Praktische stappen om dit op een nieuwe machine met GPU aan de praat te krijgen.
Twee delen: **A) inference** (DJ-transitie genereren met het pretrained model —
dit is waarschijnlijk wat je wilt) en **B) de data-pipeline** (optioneel, alleen
nodig als je eigen tracks/mixes wilt verwerken tot trainingsdata).

Vereisten op de nieuwe machine: Python 3.10+, een NVIDIA-driver die met
`nvidia-smi` werkt, en git.

```bash
nvidia-smi   # check dat de GPU + driver zichtbaar zijn voordat je begint
```

---

## A. Inference (DJ-transitie genereren)

```bash
git clone https://github.com/laudaniels/DJtrans.git
cd DJtrans

python3 -m venv .venv
source .venv/bin/activate

# torch pakt automatisch een CUDA-build die bij je driver past
pip install torch torchaudio
pip install --no-build-isolation -r requirements-inference.txt

# eenmalig: fix een bug in de madmom-library zelf (niet in onze code)
python script/patch_madmom.py
```

Systeem-dependency die madmom/pyrubberband nodig hebben (Ubuntu/Debian):

```bash
sudo apt install rubberband-cli build-essential
```

Draaien met GPU (`--n_gpu 0` = eerste GPU; weglaten of `-1` = CPU):

```bash
python script/inference.py \
  --prev_track './test/Breikthru ft Danny Devinci-Touch.mp3' \
  --next_track './test/Jameson-Hangin.mp3' \
  --prev_cue 96 --next_cue 30 \
  --n_gpu 0
```

Output komt in `results/inference/` (`..._short.wav` = alleen de transitie,
`..._full.wav` = beide tracks + transitie). Check `Using device: cuda:0` in de
output om te bevestigen dat de GPU echt gebruikt wordt.

Eigen tracks proberen: geef gewoon je eigen mp3's mee aan `--prev_track` /
`--next_track`, en zet `--prev_cue`/`--next_cue` op het punt (in seconden)
waar het vorige/volgende nummer helemaal uit-/infadet.

### GUI (drag & drop, in plaats van de command line)

Dezelfde installatie als hierboven (`requirements-inference.txt` bevat nu ook
`gradio`). Start de GUI met:

```bash
python gui/app.py --n_gpu 0
```

Dit opent een lokale webpagina (`http://127.0.0.1:7860`, link staat ook in de
terminal-output). Daar kun je:
- je twee tracks slepen/uploaden,
- het cue-point van elke track invullen (seconden waarop 'm volledig
  uit-/infadet),
- het model en device kiezen,
- op **Genereer transitie** klikken,
- de korte transitie en de volledige mix direct afspelen in de browser.

Extra opties: `--port 7860` om een andere poort te gebruiken, `--share` voor
een tijdelijke publieke link (bv. om het resultaat op je telefoon te
beluisteren — gebruik dit niet op een netwerk dat je niet vertrouwt).

### Troubleshooting A
- `ModuleNotFoundError` tijdens `pip install -r requirements-inference.txt`:
  zorg dat je `--no-build-isolation` gebruikt (madmom heeft dat nodig om de
  net geïnstalleerde cython te zien).
- Compile-errors bij madmom: ontbreekt `build-essential` (zie boven).
- `rubberband: command not found`: `sudo apt install rubberband-cli`.
- Zie je `Using device: cpu` terwijl je een GPU hebt: check `python -c
  "import torch; print(torch.cuda.is_available())"` — als dat `False` geeft,
  is het torch-CUDA-wheel niet goed geïnstalleerd (driver/CUDA-versie
  mismatch); probeer `pip install torch torchaudio --index-url
  https://download.pytorch.org/whl/cu124` (of de cu-versie die bij je driver
  past, zie `nvidia-smi` rechtsboven).

---

## B. Data-pipeline (optioneel — eigen tracks/mixes verwerken)

Alleen nodig als je zelf EDM-tracks en/of DJ-mixes met cue points hebt en daar
mixable pairs van wilt genereren (bv. om zelf te trainen). Gebruik een **apart**
venv — deze pipeline heeft TensorFlow nodig, wat niet in Deel A zit.

```bash
cd DJtrans/DJtransGAN-dg-pipeline

python3 -m venv .venv
source .venv/bin/activate

pip install torch torchaudio
pip install --no-build-isolation -r requirements-pipeline.txt

# eenmalig: fixes voor madmom/msaf (niet onze code, maar wel nodig)
python script/patch_madmom.py
python script/patch_msaf.py

# de NN-mixability-matcher: externe repo clonen + modernisering toepassen
git clone https://github.com/remyhuang/music-puzzle-games.git music_puzzle_games
python script/patch_music_puzzle_games.py
```

Eigen tracks neerzetten:

```bash
mkdir -p data/track/audio
cp /pad/naar/jouw/*.mp3 data/track/audio/
```

Pipeline draaien (volgorde is belangrijk):

```bash
# 1. features + segmenten maken uit je tracks (duurt een tijd: beat-tracking,
#    key-detectie, structuurdetectie en stem-separatie per track)
python script/create_segment.py --feature 1 --stem 1 --segment 1 --n_core 4 --n_gpu 0

# 2. mixable pairs matchen — kies een match-type:
#    rule = alleen key/bpm-regels (snel, geen GPU nodig)
#    nn   = het neurale SEN-model (preciezer, trager)
#    all  = rule-filter + nn-matching
python script/create_pair.py --match rule --n_core 4
# of:
python script/create_pair.py --match nn --n_sample 10 --n_core 1
```

Resultaat staat in `data/track/pair/` (audio + meta per gematcht paar).

Alleen relevant als je ook eigen DJ-mixes als trainingsreferentie hebt (zie
`DJtransGAN-dg-pipeline/README.md` voor het `meta.json`-formaat):

```bash
mkdir -p data/mix/audio   # + vul data/mix/meta.json met je cue points
python script/create_mix.py --n_core 4
```

### Troubleshooting B
- `ValueError: numpy.dtype size changed, may indicate binary incompatibility`:
  iets (meestal een nieuwe `pip install`) heeft numpy naar 2.x geüpgraded,
  wat de gecompileerde madmom-extensie breekt. Fix: `pip install "numpy<2"`
  opnieuw, daarna werkt het weer.
- `ImportError: cannot import name 'MutableSequence' from 'collections'` of
  `AttributeError: module 'numpy' has no attribute 'float'`: run
  `python script/patch_madmom.py` (nog) niet gedraaid.
- `ImportError: cannot import name 'inf' from 'scipy'`: run
  `python script/patch_msaf.py`.
- `AttributeError: module 'tensorflow' has no attribute 'placeholder'` bij het
  importeren van `pipeline.mixability`: `music_puzzle_games` is gecloned maar
  nog niet gepatcht — run `python script/patch_music_puzzle_games.py`.
- `create_segment.py` downloadt bij de eerste run een openunmix-separatiemodel
  (~100MB+ per stem) — dat is eenmalig en normaal, geen fout.
- Het eerste track duurt het langst (madmom/msaf zijn niet razendsnel); reken
  op zo'n 1-2 minuten per track voor feature-extractie + segmentatie.

---

## Wat al geverifieerd is (door mij, op CPU, vanaf een schone install)

- Deel A: volledige inference-run (download pretrained model → audio inladen
  → beat-tracking → BPM-matching → mixen → wav-output) werkt end-to-end.
- Deel B: feature-extractie, segment-creatie, en pair-matching met zowel
  `rule` als `nn` werken end-to-end; de NN-matcher reproduceert de originele
  referentie-output van de auteur tot ~4 significante cijfers.
- De GUI (`gui/app.py`): getest via een echte HTTP-call tegen de draaiende
  server (upload → process → audio-output), vanaf een schone install.

Wat ik **niet** heb kunnen testen (geen GPU in mijn omgeving): het daadwerkelijk
gebruiken van CUDA. De code is er klaar voor (`--n_gpu`), maar dit is het
eerste echte moment dat het op een GPU draait — hou rekening met een paar
rondes debuggen als er iets driver/CUDA-specifieks misloopt.
