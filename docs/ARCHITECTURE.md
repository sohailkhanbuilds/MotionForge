# MotionForge Architecture

## Product direction

MotionForge is a unified 3D character-animation workspace. It must own the workflow and project data rather than act as a thin collection of external API calls.

## Product principles

1. Local-first project files; cloud compute is optional and replaceable.
2. Core motion processing, retargeting, interaction, timeline and scene orchestration remain MotionForge-owned.
3. Generative models are pluggable providers. Open-source/self-hosted providers are preferred; paid APIs are optional adapters, never hard dependencies.
4. Every capability is labelled **stable**, **experimental**, or **planned**. Never imply experimental output is production quality.
5. Every processing stage has explicit inputs, outputs, configuration and diagnostics.
6. Preserve source media and non-destructive edits. Derived outputs are reproducible from project metadata where possible.
7. Low-end computers must still be able to manage projects and edit lightweight scenes; heavy inference can run remotely.

## Target architecture

- **Project layer:** project manifest, assets, source media, takes, versions and output references.
- **Application layer:** desktop UI, command interface, job queue, progress and diagnostics.
- **AI orchestration layer:** natural-language intent parsing, tool planning and guarded execution. LLMs never directly mutate arbitrary files or invent processing results.
- **Asset layer:** import, validation, metadata, materials, texture and geometry providers.
- **Character layer:** humanoid skeleton definitions, rig profiles, skinning and retarget maps.
- **Motion layer:** video decoding, body/hand/face tracking, temporal filtering, 3D reconstruction, motion solving and export.
- **Interaction layer:** object detection/tracking, contact estimation, grasp constraints, attachment and physics.
- **Scene layer:** camera, lights, environment, object transforms and scene graph.
- **Animation layer:** clips, timeline, keyframes, blending, constraints and preview.
- **Export layer:** Blender interchange first; then FBX/GLB/USD and rendered video as each exporter is validated.

## Provider boundary

Each model/provider must implement a narrow adapter contract and report:
- provider and model identifier;
- license and commercial-use constraints;
- hardware/runtime requirements;
- input/output formats;
- known limitations;
- confidence/quality diagnostics when available.

No single provider is allowed to define the project file format.

## Processing pipeline

Source video -> decode/normalize -> detect/track -> reconstruct motion -> solve skeleton -> retarget -> solve contacts/interactions -> preview/edit -> export.

Asset pipeline:

Text/image/reference -> geometry generation/import -> mesh validation -> material/texture -> rigging -> validation -> project asset.

## Current baseline (verified by source inspection, not a fresh end-to-end runtime test)

- MediaPipe single-person pose extraction.
- Confidence-weighted temporal smoothing.
- Experimental generic BVH writer.
- CSV landmarks and annotated preview.
- Experimental manually initialized 2D object tracking and Blender primitive script.

Not implemented: production-quality 3D reconstruction, auto-rigging, arbitrary-rig retargeting, face/finger capture, object geometry reconstruction, verified grasp detection, integrated scene/timeline editor, LLM assistant, desktop application.

## Immediate engineering sequence

1. Establish a durable project manifest and asset/take registry.
2. Add strict data contracts and pipeline job records.
3. Add reproducible validation fixtures and end-to-end tests.
4. Replace experimental BVH math with a validated skeleton/rotation solver.
5. Add character rig profiles and retargeting.
6. Add object tracking/contact constraints.
7. Build asset-provider adapters and optional AI orchestration.
8. Build the editor UI over stable backend contracts.
