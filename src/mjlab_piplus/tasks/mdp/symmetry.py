"""Left-right bilateral symmetry augmentation for the PiPlus robot.

The mirror is the reflection of the world through the robot's sagittal plane
(``y -> -y``). A true vector transforms as ``R = diag(1, -1, 1)``; a rotation /
angular velocity is a pseudovector and transforms as ``det(R) * R``, i.e.
``(x, y, z) -> (-x, y, -z)``. Everything below follows from those two rules.

Layout, not indices
-------------------
The old implementation sliced the actor observation with literal indices, so any
change to a term's width silently mirrored the wrong columns. Here each task
declares its observation as an ordered list of :class:`Block` s -- one per
observation term, and one per sub-command inside a composite command -- and the
global permutation / sign vectors are assembled from that. A block knows its own
width, so adding a term or changing a command's size is a one-line edit and a
mismatch against the runtime observation width raises instead of corrupting the
batch (see :meth:`Mirror.build`).

Joints
------
Every PiPlus joint mirrors as *swap sides and negate*. That holds for pitch,
roll and yaw alike because the MJCF gives each left/right pair mirrored axes
(e.g. ``r_hip_pitch`` about ``(0,-1,0)``, ``l_hip_pitch`` about ``(0,1,0)``) and
the home keyframe is antisymmetric (``l = -r``), so ``joint_pos_rel`` -- a
deviation from that pose -- negates too. Verified against the compiled spec by
``check_symmetry.py``.

Critic
------
Only the actor observation is mirrored. Under data augmentation the mirrored
sample reuses the original's value/return (``V(mirror(s)) == V(s)`` by
symmetry), so the value function is only ever evaluated on the original obs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv

__all__ = [
  "ARM_JOINTS",
  "JOINT_ORDER_12",
  "JOINT_ORDER_20",
  "LEG_JOINTS",
  "Block",
  "Mirror",
  "PiPlusGoalPoseSymmetry",
  "PiPlusMultiSymmetry",
  "PiPlusPathSymmetry",
  "PiPlusSymmetry",
  "footstep_command_block",
  "goal_pose_command_block",
  "joint_block",
  "ball_state_block",
  "kick_command_block",
  "one_hot_block",
  "path_command_block",
  "prefix_blocks",
  "repeat_block",
  "sign_block",
  "velocity_command_block",
]


##
# Joint layout.
##

LEG_JOINTS = ("hip_pitch", "hip_roll", "thigh", "calf", "ankle_pitch", "ankle_roll")
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "upper_arm", "elbow")


def _side(prefix: str, joints: Sequence[str]) -> tuple[str, ...]:
  return tuple(f"{prefix}_{name}_joint" for name in joints)


JOINT_ORDER_12: tuple[str, ...] = _side("r", LEG_JOINTS) + _side("l", LEG_JOINTS)
"""Legs only (``with_arms=False``), in MuJoCo model order: right leg, left leg."""

JOINT_ORDER_20: tuple[str, ...] = (
  _side("r", ARM_JOINTS)
  + _side("l", ARM_JOINTS)
  + _side("r", LEG_JOINTS)
  + _side("l", LEG_JOINTS)
)
"""Arms and legs (``with_arms=True``), in MuJoCo model order: right arm, left
arm, right leg, left leg."""


##
# Blocks.
##


@dataclass(frozen=True)
class Block:
  """One contiguous slice of the observation and how it mirrors.

  ``perm`` is local to the block: entry ``i`` of the mirrored block is taken
  from entry ``perm[i]`` of the original and multiplied by ``sign[i]``.
  """

  name: str
  perm: tuple[int, ...]
  sign: tuple[float, ...]

  def __post_init__(self) -> None:
    if len(self.perm) != len(self.sign):
      raise ValueError(f"Block '{self.name}': perm and sign must be the same width.")
    if sorted(self.perm) != list(range(len(self.perm))):
      raise ValueError(f"Block '{self.name}': perm must be a permutation.")

  @property
  def dim(self) -> int:
    return len(self.perm)


def sign_block(name: str, *sign: float) -> Block:
  """A block that mirrors in place, flipping only signs."""
  return Block(name, tuple(range(len(sign))), tuple(float(s) for s in sign))


def repeat_block(name: str, count: int, pattern: Sequence[float]) -> Block:
  """``count`` consecutive copies of the same in-place sign ``pattern``."""
  return sign_block(name, *(tuple(pattern) * count))


def joint_block(name: str, order: Sequence[str]) -> Block:
  """Swap each joint with its opposite-side twin and negate.

  ``order`` is the joint order of the observation (MuJoCo model order); the twin
  is found by name, so a reorder on either side can never scramble the mapping
  silently -- an unmatched joint raises.
  """
  index = {joint: i for i, joint in enumerate(order)}
  perm = []
  for joint in order:
    if joint[:2] not in ("l_", "r_"):
      raise ValueError(f"Joint '{joint}' has no l_/r_ side prefix to mirror.")
    twin = ("l_" if joint.startswith("r_") else "r_") + joint[2:]
    if twin not in index:
      raise ValueError(f"Joint '{joint}' has no mirror twin '{twin}' in the order.")
    perm.append(index[twin])
  return Block(name, tuple(perm), (-1.0,) * len(order))


##
# Observation-term blocks shared by every task.
##


def prefix_blocks(joint_order: Sequence[str]) -> list[Block]:
  """The observation prefix every PiPlus task shares.

  ``base_ang_vel | projected_gravity | joint_pos | joint_vel | actions | phase``
  -- 3 + 3 + 3J + 2 dims. Angular velocity is a pseudovector ``(-x, y, -z)``,
  gravity a true vector ``(x, -y, z)``, and the gait phase mirrors as a
  half-period shift, which on ``(sin, cos)`` is a negation of both.
  """
  return [
    sign_block("base_ang_vel", -1, 1, -1),
    sign_block("projected_gravity", 1, -1, 1),
    joint_block("joint_pos", joint_order),
    joint_block("joint_vel", joint_order),
    joint_block("actions", joint_order),
    sign_block("phase", -1, -1),
  ]


##
# Command blocks. One per command term; a MultiCommand concatenates several.
##


def one_hot_block(num_types: int) -> Block:
  """MultiCommand's active-type one-hot: the command type does not mirror."""
  return sign_block("one_hot", *([1.0] * num_types))


def velocity_command_block(stand_flag: bool) -> Block:
  """``[vx, vy, wz]`` (+ optional stand flag).

  ``vx`` is a forward velocity (unchanged), ``vy`` lateral (flips), ``wz`` a
  yaw rate (pseudovector z, flips). The stand flag is reflection-invariant.
  """
  sign = [1.0, -1.0, -1.0]
  if stand_flag:
    sign.append(1.0)
  return sign_block("velocity", *sign)


def goal_pose_command_block() -> Block:
  """``[dx_b, dy_b, cos(yaw_err), sin(yaw_err)]``: cos is even, sin is odd."""
  return sign_block("goal_pose", 1, -1, 1, -1)


def path_command_block(num_observations: int) -> Block:
  """``N`` base-frame path points ``[x, y]`` then ``[cos, sin]`` of the goal yaw
  error -- ``2N + 2`` dims, every one of them an ``[x, y]``-like pair that
  mirrors as ``[1, -1]``."""
  return repeat_block("path", num_observations + 1, (1.0, -1.0))


def footstep_command_block(num_observations: int) -> Block:
  """``M`` upcoming footsteps, 8 dims each.

  Per step: ``[foot_sign, x_b, y_b, z, roll, pitch, cos(dyaw), sin(dyaw)]``.
  The mirror swaps which foot takes the step, so ``foot_sign`` (+1 left / -1
  right) flips; ``roll`` is a pseudovector x-component and flips while
  ``pitch`` (y) does not. This is consistent with the half-period phase shift
  in the prefix, which is what swaps the swing foot in the first place.
  """
  return repeat_block(
    "footstep", num_observations, (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0)
  )


def kick_command_block() -> Block:
  """``[cos(dir_err), sin(dir_err), strength]``.

  The commanded kick direction is an angle relative to the heading, so its
  cosine is even and its sine odd; the strength is a speed.
  """
  return sign_block("kick", 1, -1, 1)


def ball_state_block() -> Block:
  """``[ball_x, ball_y, ball_vx, ball_vy]``: two base-frame planar vectors."""
  return sign_block("ball", 1, -1, 1, -1)


##
# Mirror.
##


class Mirror:
  """A permutation + sign vector assembled from an ordered list of blocks.

  The last block may be declared open-ended (``dim`` 0 with a repeating
  ``tail_pattern``), in which case its width is inferred from the observation
  the mirror is first handed. That keeps the path task's mirror correct for any
  ``num_observations`` without restating it here.
  """

  def __init__(
    self,
    blocks: Sequence[Block],
    tail_pattern: Sequence[float] | None = None,
    tail_name: str = "tail",
  ) -> None:
    self.blocks = tuple(blocks)
    self._tail_pattern = tuple(tail_pattern) if tail_pattern is not None else None
    self._tail_name = tail_name
    self._fixed_dim = sum(block.dim for block in self.blocks)
    self._perm: torch.Tensor | None = None
    self._sign: torch.Tensor | None = None
    self._cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

  @property
  def dim(self) -> int | None:
    """Total width, or ``None`` while an open-ended tail is still unresolved."""
    if self._tail_pattern is not None and self._perm is None:
      return None
    return self._fixed_dim if self._perm is None else int(self._perm.numel())

  def layout(self) -> str:
    """Human-readable block breakdown, used in the width-mismatch error."""
    parts = [f"{b.name}({b.dim})" for b in self.blocks]
    if self._tail_pattern is not None:
      parts.append(f"{self._tail_name}(n*{len(self._tail_pattern)})")
    return " | ".join(parts)

  def build(self, obs_dim: int) -> None:
    """Resolve the layout against the real observation width."""
    blocks = list(self.blocks)
    if self._tail_pattern is not None:
      remainder = obs_dim - self._fixed_dim
      stride = len(self._tail_pattern)
      if remainder < 0 or remainder % stride != 0:
        raise ValueError(
          f"Actor obs is {obs_dim} dims; the mirror expects "
          f"{self._fixed_dim} + a multiple of {stride} ({self.layout()})."
        )
      blocks.append(
        repeat_block(self._tail_name, remainder // stride, self._tail_pattern)
      )
    total = sum(block.dim for block in blocks)
    if total != obs_dim:
      raise ValueError(
        f"Actor obs is {obs_dim} dims but the mirror covers {total}: "
        f"{self.layout()}. Update the block list in symmetry.py to match the "
        "observation terms."
      )
    perm: list[int] = []
    sign: list[float] = []
    offset = 0
    for block in blocks:
      perm.extend(offset + p for p in block.perm)
      sign.extend(block.sign)
      offset += block.dim
    self._perm = torch.tensor(perm, dtype=torch.long)
    self._sign = torch.tensor(sign, dtype=torch.float32)
    self._cache.clear()

  def _tensors(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if self._perm is None:
      self.build(ref.shape[-1])
    assert self._perm is not None and self._sign is not None
    if int(self._perm.numel()) != ref.shape[-1]:
      # The layout was resolved against a different width: rebuild and let the
      # size check above report the mismatch.
      self._perm = None
      self.build(ref.shape[-1])
      assert self._perm is not None and self._sign is not None
    cached = self._cache.get(ref.device)
    if cached is None:
      cached = (
        self._perm.to(ref.device),
        self._sign.to(device=ref.device, dtype=ref.dtype),
      )
      self._cache[ref.device] = cached
    return cached

  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    perm, sign = self._tensors(x)
    return x[..., perm] * sign


##
# Symmetry callables.
##


class PiPlusSymmetry:
  """Callable left-right symmetry function for the PiPlus robot.

  Passed as ``data_augmentation_func`` to rsl_rl's ``Symmetry`` extension.
  Returns (obs, actions) with the original batch prepended to the mirrored
  batch along dim 0 (num_aug = 2).

  The default layout is the velocity task: the 12-DoF prefix followed by a
  4-dim ``[vx, vy, wz, stand_flag]`` twist command.
  """

  def __init__(
    self,
    obs_mirror: Mirror | None = None,
    joint_order: Sequence[str] = JOINT_ORDER_12,
  ) -> None:
    self.obs_mirror = obs_mirror if obs_mirror is not None else self._default_mirror()
    self.action_mirror = Mirror([joint_block("actions", joint_order)])

  @staticmethod
  def _default_mirror() -> Mirror:
    return Mirror(
      [*prefix_blocks(JOINT_ORDER_12), velocity_command_block(stand_flag=True)]
    )

  @torch.no_grad()
  def __call__(
    self,
    env: VecEnv,
    obs: TensorDict | torch.Tensor | None,
    actions: torch.Tensor | None,
  ) -> tuple[TensorDict | torch.Tensor | None, torch.Tensor | None]:
    if obs is None:
      obs_aug = None
    elif isinstance(obs, TensorDict):
      actor = cast(torch.Tensor, obs["actor"])
      batch_size = int(obs.batch_size[0])
      obs_aug = obs.repeat(2)
      aug_actor = cast(torch.Tensor, obs_aug["actor"])
      aug_actor[:batch_size] = actor
      aug_actor[batch_size:] = self.obs_mirror(actor)
    else:
      obs_aug = torch.cat([obs, self.obs_mirror(obs)], dim=0)

    if actions is None:
      action_aug = None
    else:
      action_aug = torch.cat([actions, self.action_mirror(actions)], dim=0)

    return obs_aug, action_aug


class PiPlusGoalPoseSymmetry(PiPlusSymmetry):
  """Goal-pose task: 12-DoF prefix + ``[dx, dy, cos(yaw_err), sin(yaw_err)]``."""

  @staticmethod
  def _default_mirror() -> Mirror:
    return Mirror([*prefix_blocks(JOINT_ORDER_12), goal_pose_command_block()])


class PiPlusPathSymmetry(PiPlusSymmetry):
  """Path-following task: 12-DoF prefix + the path command's ``2N + 2`` tail.

  The tail is left open-ended and sized from the runtime observation, so
  changing ``PathCommandCfg.num_observations`` needs no edit here. The task is
  left-right symmetric: reflecting the path across the robot's forward axis
  together with a mirrored policy yields mirrored behavior with identical
  reward (path distance and heading alignment are both reflection-invariant).
  """

  @staticmethod
  def _default_mirror() -> Mirror:
    return Mirror(
      prefix_blocks(JOINT_ORDER_12), tail_pattern=(1.0, -1.0), tail_name="path"
    )


class PiPlusMultiSymmetry(PiPlusSymmetry):
  """Multi-command task: 20-DoF prefix + the composite command.

  The command tail is
  ``[one_hot(6) | velocity(3) | path(2N+2) | footstep(8M) | goal_pose(4) |
  kick(3)]`` -- ``MultiCommand.command`` emits the one-hot first and then each
  sub-command in ``terms`` order, skipping the zero-width ``stand``. Blocks are
  fully explicit here (no open-ended tail) because the sizes of two different
  sub-commands would otherwise be indistinguishable from the total width.

  The ball observation is a term of its own, after the command, so it is the
  last block.
  """

  def __init__(
    self,
    num_types: int = 6,
    num_path_observations: int = 8,
    num_footstep_observations: int = 4,
  ) -> None:
    super().__init__(
      obs_mirror=Mirror(
        [
          *prefix_blocks(JOINT_ORDER_20),
          one_hot_block(num_types),
          velocity_command_block(stand_flag=False),
          path_command_block(num_path_observations),
          footstep_command_block(num_footstep_observations),
          goal_pose_command_block(),
          kick_command_block(),
          ball_state_block(),
        ]
      ),
      joint_order=JOINT_ORDER_20,
    )

    
