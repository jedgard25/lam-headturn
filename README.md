# lam-headturn

Single photo in, a head you can turn out: a 2.5D "head-turn" avatar rendered as Gaussian splats in the browser, with the head following your cursor.

This started as an experiment to get 2.5D-style headshots from one photo, **without FLAME**. It runs [LAM-20K](https://huggingface.co/3DAIGC/LAM-20K) on-device via CoreML, then reshapes and recolours LAM's splats so they match the photo. It is an experiment and still has artifacts (see [docs/NOTES.md](docs/NOTES.md)).

macOS / Apple Silicon only: it uses CoreML and Apple Vision (via the small Swift tools).

## How it works

1. LAM-20K (CoreML) produces 20,018 splats from a 518px crop. Positions come from a fixed canonical head grid, and since the FLAME template is dropped, head shape isn't tied to the photo.
2. `refine.py`: Apple Vision landmarks and a thin-plate warp pull the splats onto the photo's face and silhouette; Depth Anything V2 Small adds depth.
3. `fit.py`: a differentiable fit adjusts a smooth 2D warp, then colours, against the photo.
4. `clean.py`: removes floaters and caps splat size.
5. `index.html`: WebGL splat viewer. `server.py` serves it and exposes `POST /reconstruct`.

Full design notes, numbers, and what didn't work: [docs/NOTES.md](docs/NOTES.md).

## Setup

```bash
uv venv && uv pip install -r requirements.txt   # or python -m venv + pip

# build the Swift helpers (Vision landmarks + background matte)
swiftc -O tools/landmarks.swift -o tools/landmarks
swiftc -O tools/matte.swift -o tools/matte

# weights: INT8 CoreML model from Hugging Face
huggingface-cli download spizzerp/LAM-20K-CoreML --local-dir model
python dequant.py        # writes model/LAMReconstruct_fp16.mlpackage
```

Why the dequant step: the INT8 package segfaults on the GPU compute path. `dequant.py` rewrites it as fp16 (about 1.2 GB). It adds no precision beyond the INT8 weights; it only makes the model runnable on GPU. The fp16 package is therefore not checked in.

## Run

```bash
python server.py            # http://localhost:8000
python reconstruct.py photo.jpg -o out.ply   # headless, raw LAM only
python evalloop.py photo.jpg -o sheet.png --variants nofit,old,new   # eval contact sheet
```

Use a roughly frontal photo with one face. Latency is around 7 s for raw LAM plus about 11 s for the fit.

## Licensing (unresolved)

- LAM weights: the HF card says Apache-2.0, the upstream GitHub repo says CC-BY-NC. Treat the weights as non-commercial until that is clarified.
- Depth Anything V2 **Small** is Apache-2.0 (Base/Large are CC-BY-NC).
- No FLAME assets are used or included.
- No license is set on this repo's own code yet.
