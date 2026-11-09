"""Retarget canonical pose sequences using a saved target rig profile."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .retarget_sequence import retarget_sequence
from .rig import RigProfile
from .skeleton import JOINTS, SkeletonError


def retarget_to_profile(
    source_frames: Sequence[Mapping[str, np.ndarray]],
    profile: RigProfile,
    root_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Fit canonical source poses to a saved rig's bone lengths.

    Output is (frames, canonical_joints, xyz) in JOINTS order. This does not
    bind the result to a mesh or produce skinning/animation files.
    """
    if not isinstance(profile, RigProfile):
        raise SkeletonError("profile must be a validated RigProfile.")
    result = retarget_sequence(source_frames, profile.bone_lengths, root_positions)
    if not np.all(np.isfinite(result)):
        raise SkeletonError("Retargeting produced non-finite coordinates.")
    return result


def sequence_to_joint_maps(frames: np.ndarray) -> list[dict[str, np.ndarray]]:
    """Convert canonical (N, joints, 3) arrays to retargeter input mappings."""
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(JOINTS), 3):
        raise SkeletonError(
            f"frames must have shape (N, {len(JOINTS)}, 3) in canonical joint order."
        )
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise SkeletonError("frames must be non-empty and finite.")
    return [
        {joint.name: values[frame_index, joint_index].copy()
         for joint_index, joint in enumerate(JOINTS)}
        for frame_index in range(len(values))
    ]
