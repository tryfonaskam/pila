# PILA macOS / Apple Silicon Edition

This repository contains a macOS- and Apple Silicon–compatible implementation
of the PolyTrack Imitation Learning Agent (PILA).

This work is adapted from the original PILA project:
https://github.com/tryfonaskam/pila

The goal of this repository is to provide a stable macOS workflow
(screen capture, training, and inference) without affecting the main codebase.

## Key Differences

- macOS-safe screen capture backend
- Apple Silicon (MPS) training support
- Stable long-running training on macOS laptops

## Quick Start (macOS)

Environment setup:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Record data:
```
python -m capture.datasetmake
```

Train:
```
python train.py
```

Run agent:
```
python play.py
```

## Credits

- Original PILA project by **[tryfonaskam](https://github.com/tryfonaskam)**
- macOS / Apple Silicon adaptation by **[sahusaurya](https://github.com/sahusaurya)**
