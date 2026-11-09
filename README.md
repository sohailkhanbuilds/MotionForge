# MotionForge

AI-assisted, markerless human motion capture for Blender.

**Pipeline:** recorded video -> MediaPipe 3D pose landmarks -> temporal smoothing -> experimental BVH export -> Blender.

> MotionForge 0.1 is an experimental single-camera MVP, not a professional optical motion-capture system. It estimates joint pose from MediaPipe world landmarks and estimates root travel from 2D image position. Occlusion, depth ambiguity, foot sliding and imperfect retargeting are expected.

## MVP features

- Process recorded video (formats supported by OpenCV).
- Detect one human body with MediaPipe Pose.
- Extract 3D hip-centred world landmarks plus normalized image landmarks.
- Confidence-weighted Gaussian smoothing to reduce jitter while preserving movement.
- Use a robust short-window reference pose and confidence-gate unreliable joints during BVH export.
- Export a generic humanoid BVH and landmark CSV.
- Render an annotated MP4 pose-tracking preview for visual quality checks.
- Report per-frame and body-region detector visibility in metadata.
- Colab notebook for remote execution; no paid API key.

## Requirements

- Python 3.10 or 3.11.
- 8 GB RAM recommended. Your 4 GB PC may run short clips with the light model, but installation and processing may be slow.
- Blender 3.6+ to inspect BVH.

## Install

```bash
git clone https://github.com/sohailkhanbuilds/MotionForge.git
cd MotionForge
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python -m motionforge --input path/to/video.mp4 --output-dir output
```

Use the lightest model and short test clips on low-memory PCs:

```bash
python -m motionforge --input walk.mp4 --output-dir output --model-complexity 0 --smooth-window 5
```

Outputs:
- `output/motionforge.bvh` - experimental BVH animation.
- `output/landmarks.csv` - per-frame world/image landmarks and visibility.
- `output/metadata.json` - source details and known limitations.

## Import into Blender

1. Open Blender and enable/install its BioVision Motion Capture (BVH) importer if it is not already available.
2. Use **File -> Import -> Motion Capture (.bvh)**.
3. Select `motionforge.bvh`.
4. Inspect the armature and Action in the timeline.
5. Retarget the generic skeleton to your character separately. Automatic mapping to arbitrary custom rigs is not implemented yet.

## Google Colab

Open [MotionForge_Colab.ipynb](notebooks/MotionForge_Colab.ipynb) in Colab and run all cells. **Important:** this MVP uses MediaPipe's standard Python Pose solution and does not currently execute inference on the Colab GPU. Colab is a remote runtime, not GPU acceleration for this implementation.

## Capture tips

- Keep the full body visible, including feet and hands.
- Use a stable camera, bright even lighting and minimal motion blur.
- Record one person at a time; begin with a 5-10 second clip.
- A side-on camera is useful for testing screen-plane movement.

## Known limitations

- Single-camera 3D estimates are approximate.
- MediaPipe world landmarks are hip-centred. Root travel is approximated from normalized 2D image position using a rough scale; true depth and calibrated world translation are not recovered.
- Fast movement and occluded limbs reduce accuracy.
- Fingers and facial expressions are not captured.
- BVH hierarchy and rotations are experimental. Expect cleanup; foot contact and retargeting are not solved.
- This is offline video processing, not live low-latency capture.
- Colab sessions can disconnect. GPU acceleration is not implemented in this version.

## Roadmap

- [x] Recorded-video pose extraction
- [x] Temporal smoothing
- [x] Experimental BVH and CSV export
- [ ] Validate BVH against varied clips and Blender import
- [ ] Blender retargeting helper
- [ ] Better root trajectory and foot contact reconstruction
- [ ] Live capture and latency controls
- [ ] GPU-accelerated inference backend

## License

MIT. See [LICENSE](LICENSE).


## Optional object tracking (experimental)

Track one visible object by selecting a tight first-frame bounding box in pixel coordinates:

    python -m motionforge --input clip.mp4 --output-dir output --object-roi 120 80 90 90 --object-type ball

Supported primitive types: ball, box, cube, cylinder, bottle, cone. This exports object_motion.json and object_animation.py. Open Blender, switch to the Scripting workspace, open and run object_animation.py to create a primitive with keyframed screen-plane motion.

Limitations: the object ROI is manually selected; tracking can drift or fail during occlusion; geometry is a simple primitive; depth is not reconstructed; hand proximity is only a heuristic and does not create a verified grasp/parent constraint. The object is not yet automatically attached to the animated character rig.
