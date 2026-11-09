# MotionForge Development Roadmap

This roadmap is a sequence of measurable engineering gates, not a promise that every advanced feature already works.

## Gate 0 — Project foundation
- [x] Existing experimental pose/BVH/object prototype inventoried.
- [x] Provider-agnostic project manifest and asset/take registry started.
- [ ] Project format migration/versioning tests.
- [ ] Job model, progress events and reproducible run metadata.

**Exit criteria:** projects can be created, reopened, validated and moved without losing asset/take references.

## Gate 1 — Motion core reliability
- [ ] Validate BVH hierarchy, offsets, rotation order and coordinate conventions in Blender.
- [ ] Add calibrated skeleton definition and unit tests for rotations.
- [ ] Add missing-joint handling and foot-contact diagnostics.
- [ ] Compare output across front/side/occlusion test clips.

**Exit criteria:** documented accuracy limits and repeatable output on a test set.

## Gate 2 — Character pipeline
- [ ] Character import (GLB/FBX) and skeleton inspection.
- [ ] Rig profile schema and bone mapping.
- [ ] Retargeting solver with rest-pose calibration.
- [ ] Animation preview and export validation.

## Gate 3 — Object and interaction pipeline
- [ ] Replace single-box tracker with detector + temporal tracker adapter.
- [ ] Track camera and object scale where possible.
- [ ] Contact/grasp state estimation with confidence and manual correction.
- [ ] Hand attachment constraints and release/throw transitions.
- [ ] Primitive proxy first; generated geometry remains a separate asset stage.

## Gate 4 — Asset generation and AI orchestration
- [ ] Provider-neutral interfaces for image/text-to-3D and auto-rigging.
- [ ] Prefer self-hostable/open-weight models; document model licenses and GPU requirements.
- [ ] Qwen-family LLM assistant for intent parsing and safe tool planning.
- [ ] Human approval for destructive operations and expensive cloud jobs.

## Gate 5 — Integrated studio
- [ ] Desktop project browser and 3D viewport.
- [ ] Timeline, clip library, keyframe editing and scene graph.
- [ ] Job queue, progress, cancellation and diagnostics.
- [ ] Export profiles and packaging.

## Resource strategy
The target development PC (Core 2 Duo, 4 GB RAM, Intel G41) cannot run modern 3D generation models locally. Keep editing/project management lightweight and make inference backends optional: local CPU for small tasks, Colab/cloud GPU for heavy models. No paid API should be required for the baseline workflow.
