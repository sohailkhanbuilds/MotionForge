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
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Run a short clip:

```bash
python -m motionforge --input path/to/video.mp4 --output-dir output --model-complexity 0
```

Outputs include experimental BVH, landmarks CSV, metadata and annotated pose preview.

## Blender import bridge (experimental)

MotionForge can generate a standalone Python script for Blender to import a BVH take and save a `.blend` scene. Blender is an optional external application; generating the script does not execute Blender.

```bash
python -m motionforge blender-script --bvh output/motion.bvh --script output/import_take.py --blend-output output/scene.blend
```

Run the generated script with Blender:

```bash
blender --background --python output/import_take.py
```

Optionally import a character asset:

```bash
python -m motionforge blender-script --bvh output/motion.bvh --script output/import_take.py --blend-output output/scene.blend --character assets/hero.glb
```

To map BVH channels onto a character armature, pass `--target-armature HeroRig`. By default, identical bone names are mapped automatically. For different naming conventions, create a JSON file whose keys are BVH source bone names and whose values are target rig bone names, then pass `--bone-map config/bone-map.json`:

```json
{
  "Hips": "pelvis",
  "Spine": "spine_01",
  "LeftArm": "upper_arm.L",
  "RightArm": "upper_arm.R"
}
```

The bridge bakes mapped local quaternion rotations into a new target action and copies root location. This is an **experimental local-channel transfer**, not full skeletal retargeting: it does not correct rest-pose differences, bone roll/axes, scale, proportions or IK constraints. Unmapped bones remain untouched. A wrong map can produce distorted motion, so inspect the result in Blender. Without a target armature, the character and BVH armature are imported separately.

Blender runtime compatibility must be tested with the user's installed Blender version and actual rig. The Python tests validate generated script syntax and configuration guards, not Blender's import operators or final animation quality.

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


## Retargeted BVH export and validation (experimental)

A canonical retargeted sequence can be adapted to the current generic BVH hierarchy:

```python
from motionforge.canonical_bvh import export_canonical_bvh
from motionforge.bvh_validate import validate_bvh

export_canonical_bvh(retargeted_frames, fps=30, output_path="exports/hero.bvh")
report = validate_bvh("exports/hero.bvh")
print(report)
```

The validator checks hierarchy/channel declarations, frame count, frame time, row width and finite numeric values. It validates MotionForge's current BVH dialect, not every third-party BVH variant. The canonical adapter currently exports the supported 16-joint skeleton; it is not a full skeletal mesh binding or production-grade anatomical rig. The current head/neck mapping is simplified and needs a dedicated hierarchy/rotation pass before claiming high-fidelity character animation.


### BVH head/neck and input checks

The current BVH writer now estimates the neck from shoulders plus the nose landmark, while the head centre is estimated independently from the two ear landmarks. The canonical retarget adapter maps its head joint to those ear positions. This is a more useful separation than reusing the nose for both joints, but it remains a coarse landmark-derived head pivot, not a full skull/neck anatomical rig. Export FPS is now checked for finite values in the range 0–240.
