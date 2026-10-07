# LAM head-turn: status notes

## Start
- LAM-20K CoreML (`model/LAMReconstruct_fp16.mlpackage`) -> 20018 splats (14 ch: pos3 rgb3 op1 scale3 quat4), `reconstruct.py` / `server.py`, WebGL viewer `index.html` (yaw follows cursor).
- Positions are a fixed canonical head grid (index = semantic location, e.g. eyes/lips), NOT image-aligned. FLAME base was dropped (licensing), so head shape/aspect isn't tied to the photo. Result: ~40/60, narrow/tall heads.
- Viewer `splat size` defaulted 1.5 (inflates every splat -> blur, chin halo). Now 1.0.

## Pipeline now (`POST /reconstruct?refine=1&fit=1&zoom=1`)
1. LAM infer on 518 crop (browser does Vision matte + flat-bg composite first).
2. `refine.py` warp: Apple Vision landmarks (`tools/landmarks.swift`, built to `tools/landmarks`) on photo vs. a fast frontal render of the splats -> thin-plate warp of splat xy; plus silhouette rows (hair/head outline from flat-bg mask), splat scale follows warp Jacobian, z scaled by width^0.3.
3. Depth: Depth Anything V2 Small (Apache-2.0, HF `depth-anything/Depth-Anything-V2-Small-hf`, torch/MPS) calibrated to LAM z by robust regression, 60% blend over face polygon.
4. `fit.py` photo fit = `shape_fit` (`fit_iters=200`, ~8-10 s): fits only what one frontal photo constrains: a smooth 2D warp (32x32 control grid of screen-space offsets) + a very low-frequency colour gain (6x6 grid, +-25%), against the photo and its silhouette mask, through `comp_render` (sorted alpha-over, same blend as the viewer; 259px). Depth, opacity and per-splat colour are untouched.
   - Replaced the per-splat fit (`fit.optimize`, still in the file for A/B). It optimised through a depth-weighted-average renderer, so (a) z became a "make my colour win" knob -> floaters (lip highlight), (b) colours/opacity were tuned for the wrong blend -> saturated sweater, orange hair, (c) free per-splat motion + neighbour-smoothed colour -> washed-out hair.
5. `clean.py`: hide floaters, pull strays to local plane, cap splat size at 2x local spacing.
- Other files: `splatrender.py` (exact torch renderer matching the viewer camera; slow, eval only), `samples/headshot.jpg` (test), `out.ply` (last headshot result), `out_raw_backup.ply` (original).
- Viewer: checkboxes `depth refine`, `photo fit`; idle frames are skipped (fans).

## Numbers (mean |render - photo|, frontal, exact renderer)
- headshot (`evalloop.py`, 384px, refine camera): warp only 0.054, old per-splat fit 0.064, shape_fit 0.038.
- (older numbers, different harness) headshot: raw LAM 0.119 -> warp 0.066 -> +fit 0.049. man photo: warp 0.034.
- Latency warm: raw LAM ~7 s (4-10 s depending on GPU contention, e.g. open viewer tabs), warp +0.5 s, fit +~11 s.

## Renderer bug (fixed) - read this first
- `splatrender.render` and `fit.project` built the screen-space covariance with y up but used it with y down, so every splat's orientation was mirrored (radial streaks, broken glasses) vs. the WebGL viewer. Everything that went through them was affected: the landmark render, both fits, and every "exact renderer" number in this file from before the fix. Check with a viewer screenshot of out.ply vs `sr.render` of the same file (Playwright MCP works for the screenshot).
- After the fix on headshot (evalloop, 384px): old per-splat fit 0.035, shape_fit 0.035, shape_fit + color_fit 0.027.

## Head shape: LAM's output is a hood, not a head
- Plot the splat centres from the side/top (no render needed): raw LAM is an open front shell. It fans out toward the back like a camera frustum, there is no back of the head, and its rim (= the hair outline) hangs ~0.2 units behind the nose, about twice too deep. Turning/tilting swings that rim out: "head bigger in the back", looming forehead/crown.
- `sr.unflare()` runs before refine (and on the raw path in server.py): (1) squash depth behind the face surface x0.55, (2) re-project from a close camera (`SRC_K` = 3.2 radii, the old viewer camera) to ours, which keeps the frontal picture and makes the sides parallel.
- Camera is now one shared portrait lens: `sr.FOV = .3`, `sr.DIST_K`, mirrored as `FOV` / `DISTK` in index.html (keep in sync). Changing the camera alone, without unflare, made it look worse: the close camera had been hiding the flare.
- Checked on the headshot only (frontal err 0.024, yaw +-35 looks like a normal head). Knobs if other photos disagree: `squash` (lower = flatter back), `SRC_K`, `FOV`.

## Colour from the photo
- `fit.color_fit` runs after `shape_fit`: geometry/opacity frozen => the frontal image is linear in splat colours, so blend weights are computed once (sorted alpha-over, 518px) and colours fitted to the photo (~1 s). Gives the person's real eyes (iris colour, lid line, eyeliner) instead of LAM's generic wide-open ones.
- L2 pull back to LAM colour, stronger for big splats (`area_pow`) and faint ones (`op_pow`): without the opacity term the glasses lenses go orange and show from the side; without the area term big soft splats blotch.
- `evalloop.py` variants: `new@c_iters=0` = shape only; `new@c_lam=0.1;c_op_pow=1.0` etc.

## Tilted photos
- Vision's `roll` is always 0 for the landmark request we use, so the old de-roll never ran and the warp bent LAM's upright canonical head into the tilted face (droopy/uneven eyes). `refine()` now levels the photo by the eye line first (rotating about the face centre, bg-colour fill, `valid` mask for the fit) and fits everything to that; avatar comes out upright.
- `shape_fit(anchors=...)` holds the fit warp near zero at the landmark points (it was pulling an eye ~5 px sideways).
- `evalloop.py`'s photo column / err are against the un-levelled photo, so err is not meaningful for tilted inputs.

## Eval loop
- `python evalloop.py samples/headshot.jpg -o sheet.png --variants nofit,old,new` -> contact sheet (photo | front | error x3 | yaw -/+35 | face crop) with the exact renderer; `new@gain_bound=0.0;grid=24` tries shape_fit settings. Render with the camera refine used (`sr.camera(raw LAM g)`), not the refined splats' own, or the error is meaningless.
- Gain grids finer than ~10 put an orange tint inside the glasses lenses.

## Findings
- Landmark error vs photo: 23 px -> ~2 px after warp.
- Fit is what makes the front view match (uncanny without it) but bakes frontal view into splats: at large yaw (>~35 deg) eyes/eyebrows slide against skin (feature and skin layers at different depth), texture flatter.
- Remaining artifacts: hair "horn"/spikes at crown, ears stretched pointy after warp, glasses streaks (native LAM), blotchy neck/sweater, side hair looks like a smeared wall.

## Tried, didn't help (reverted or off)
- Side-view consistency loss in fit (`side_yaws`): helps a bit, 96 s -> unusable.
- Light position-only fit; low-frequency color gain (`fit.color_gain`): washed out detail / worse error on man.
- Depth weight 0.9: washed-out eyes. Surface merge (`merge=` in refine, median z over screen neighbors): face breaks apart (eyes pushed behind skin). Left in code, default 0.
- Per-axis splat scale fitting (`opt_scale`): no gain, blurrier. Off.

## Left / ideas
- Depth-aware fit: constrain features and underlying skin to share depth (the real fix for the yaw sliding).
- Ear fix: limit warp influence near ears. Clamp hair z / crown spikes.
- Test depth weight higher with yaw check; surface-aligned (surfel) splat orientation from depth normals for skin.
- Normals model (DSINE / Sapiens; check licenses) instead of / in addition to depth. Novel-view model for pseudo side views (heavy; license/speed).
- Cap yaw to ~20-25 deg as the cheap mitigation.
- Run on more faces (beards, glasses, off-axis, non-flat bg: fit is skipped when bg isn't flat; no-face -> raw LAM silently).
- Licensing: LAM weights (HF Apache-2.0 vs GitHub CC-BY-NC) unresolved; Apple Vision is macOS/iOS-only; DA2 Base/Large are CC-BY-NC (Small is Apache-2.0).

## Depth mesh (viewer) and why not triangulate the splat centres
- `index.html` bakes the splats' frontal render (colour + coverage-weighted expected depth, 768px, RGBA16F) and draws a 256x256 grid displaced by that depth (`mesh blend` slider mixes it with the splats). No edge-on shards; the depth map is mip-blurred (lod 3.5 in `mvs`) to hide feature jitter.
- Point-triangulated mesh (`mesh from splat points`, `faces.bin`, `tools/make_faces.py`): Delaunay of the splat centres in frontal xy, topology made once from the raw canonical grid (splat index = face location) and reused for any photo; vertex xy/uv from the frontal bake, vertex depth = repeated median over mesh neighbours. First attempts had holes: (a) the canonical cloud is layered (eyes, lips, lashes, hair float over skin), so xy-neighbours are not surface neighbours; a 3x-median edge filter then deleted ~9k face triangles; (b) ~half the splats (10.6k/20k) have opacity < .15 and pollute the triangulation. Fix: triangulate only splats with opacity > .15 (16k triangles), keep every triangle, median depth. Result: no holes; remaining: faceting in hair, a small gap near the brow/ear.
