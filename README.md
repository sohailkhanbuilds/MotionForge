# MotionForge

AI-assisted, markerless human motion capture for Blender.

**Pipeline:** recorded video -> MediaPipe 3D pose landmarks -> smoothing -> BVH export -> Blender.

> MotionForge 0.1 is an experimental single-camera MVP. It estimates motion from MediaPipe world landmarks. It is not a professional optical motion-capture system: occlusion, depth ambiguity, foot sliding and imperfect retargeting are expected.

## MVP features

- Process a recorded video (MP4/MOV/AVI supported by your OpenCV build).
- Detect one human body with MediaPipe Pose.
- Extract 3D world landmarks.
- Smooth landmark trajectories.
- Export a BVH skeleton animation.
- Export landmark CSV for debugging.
- Optional Colab notebook for cloud execution.
- No paid API key required.

## Requirements

- Python 3.10 or 3.11 recommended.
- 8 GB RAM recommended; 4 GB may work with small/short videos but can be slow.
- Webcam is not required for the first version; use a recorded video.
- Blender 3.6+ for importing and inspecting BVH.

## Install

```bash
git clone https://github.com/sohailkhanbuilds/MotionForge.git
cd MotionForge
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python -m motionforge --input path/to/video.mp4 --output-dir output
```

Optional settings:

```bash
python -m motionforge --input walk.mp4 --output-dir output --model-complexity 0 --smooth-window 5
```

Outputs:
- `output/motionforge.bvh` - BVH animation.
- `output/landmarks.csv` - per-frame 3D landmark positions and visibility.
- `output/metadata.json` - source FPS, frame count, capture settings and notes.

### Import into Blender

1. Open Blender.
2. Choose **File -> Import -> Motion Capture (.bvh)**. If the menu is unavailable, enable Blender's built-in **BioVision Motion Capture (BVH)** importer in Preferences -> Get Extensions/Add-ons (wording varies by Blender version).
3. Select `motionforge.bvh`.
4. Inspect the armature in the Action Editor and play the timeline.
5. Retarget to your character using a compatible retargeting workflow/add-on. The generated skeleton is a generic humanoid, not automatically mapped to every custom rig.

## Google Colab

Open `notebooks/MotionForge_Colab.ipynb` in Google Colab, upload a video and run the cells. Select a GPU runtime if available. The pose model can also run on CPU; GPU availability is not guaranteed.

## Capture tips

- Keep the full body visible, including feet and hands.
- Use a stable camera and bright, even lighting.
- Avoid loose clothing and strong motion blur.
- Record one person at a time.
- Use a 5-20 second test clip first.
- Keep the camera still for the first tests.

## Current limitations

- Single-camera 3D estimates are approximate.
- Fast movement and occluded limbs reduce accuracy.
- Hands/fingers and facial expressions are not captured.
- BVH export uses a generic joint hierarchy and approximate joint rotations; expect cleanup and retargeting.
- This MVP is offline video processing, not live low-latency capture.
- A Colab session may disconnect and free GPU resources.

## Roadmap

- [x] Repository MVP foundation
- [x] Recorded-video pose extraction
- [x] Smoothed landmark export
- [x] Experimental BVH writer
- [ ] Validate BVH against varied real-world clips
- [ ] Blender retargeting helper
- [ ] Multi-person / hand tracking
- [ ] Live capture with latency controls
- [ ] Better foot contact and motion reconstruction

## License

MIT. See [LICENSE](LICENSE).
