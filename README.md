# MotionForge

AI-assisted, markerless human motion capture and 3D animation workspace.

**Current baseline:** recorded video -> MediaPipe pose landmarks -> temporal smoothing -> experimental BVH export. Project management is being built as a provider-neutral foundation.

> MotionForge is under active development. The existing motion/BVH/object outputs are experimental, not professional motion-capture quality.

## Project management

Create a project:

```bash
python -m motionforge project init ./MyScene --name "My Scene"
```

Register a character or prop asset (the file path is relative to the project):

```bash
python -m motionforge project add-asset ./MyScene --name Hero --type character --file assets/hero.glb
```

Register a motion take:

```bash
python -m motionforge project add-take ./MyScene --name Walk --video source/walk.mp4 --motion takes/walk.bvh
```

Inspect the project:

```bash
python -m motionforge project info ./MyScene
```

The project manifest is `motionforge.project.json`. It stores asset and take references, not large video or mesh binaries. Place files in the matching project folders.

## Existing motion capture pipeline

Requirements: Python 3.10 or 3.11; 8 GB RAM recommended. A 4 GB PC may run short clips with the light model, but installation and processing may be slow.

```bash
git clone https://github.com/sohailkhanbuilds/MotionForge.git
cd MotionForge
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run a short clip:

```bash
python -m motionforge --input path/to/video.mp4 --output-dir output --model-complexity 0
```

Outputs include experimental BVH, landmarks CSV, metadata and annotated pose preview.

## Object tracking (experimental)

A user-selected first-frame ROI can be tracked with OpenCV and exported as a simple Blender primitive animation. This is 2D screen-plane tracking only; it does not reconstruct object geometry, true depth or verified hand grasping.

## Architecture and roadmap

- [Architecture](docs/ARCHITECTURE.md)
- [Development roadmap](docs/ROADMAP.md)

The core motion solver, retargeting, interaction engine and editor are not yet production-ready. Model providers are intended to be replaceable adapters; paid APIs are not required for the core project format or motion pipeline.

## Google Colab

Open [MotionForge_Colab.ipynb](notebooks/MotionForge_Colab.ipynb). The current MediaPipe implementation uses CPU inference; a Colab GPU runtime does not automatically accelerate it.

## License

MIT. See [LICENSE](LICENSE).


## Target rig profiles (experimental)

Create a reusable canonical humanoid bone-length profile from a clean reference pose:

```python
from motionforge.rig import RigProfile

profile = RigProfile.from_pose("Hero", reference_frame)
profile.save("Hero.rig.json")
```

Load it later with `RigProfile.load(...)` and pass its `bone_lengths` mapping to the sequence retargeter. Profiles currently describe canonical bone lengths in metres only; they do not create mesh skeletons, bind poses, skin weights or guarantee compatibility with arbitrary GLB/FBX rigs.


Apply a saved profile to a canonical pose sequence:

```python
from motionforge.rig import RigProfile
from motionforge.rig_retarget import retarget_to_profile

target = RigProfile.load("Hero.rig.json")
retargeted = retarget_to_profile(source_joint_frames, target)
```

The result is canonical joint-coordinate data, not a skinned character or a BVH/GLB export. Mesh binding and production-quality joint rotations remain future work.
