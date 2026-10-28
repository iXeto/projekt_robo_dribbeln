"""Multi-command env: one policy driven by a velocity, path, footstep, goal-pose,
chris_dribble or stand command.

A single ``MultiCommand`` assigns each env one of three command types and
switches between them during the episode, so the policy learns the hand-over
between modes rather than being three separate policies.

The policy sees every sub-command concatenated with the inactive blocks zeroed,
prefixed by a one-hot naming the active type:

    [one_hot(6) | velocity(3) | path(2N + 2) | footstep(8M) | goal_pose(4)
     | chris_dribble(3)]

The ball's position and velocity are *not* in there: they are a sensed quantity
on a camera's timescale, so they ride in their own observation term with a much
longer delay, zeroed whenever the chris_dribble command is not the active type.

Standing is its own command type here, not a flag on the velocity command, so a
zero-velocity *velocity* command means walk-in-place. Reward terms that only
apply to some command types carry ``command_weights``; terms without the field
apply at full weight everywhere.

Fall recovery is folded in rather than being a command of its own: a fraction of
envs respawn on the floor, there is no fall termination, and the getup terms are
live under every command type. See ``mdp/getup.py`` and the "Getup" constants
below. This is also the only task that actuates the arms (20 DoF), so its
left-right mirror is ``PiPlusMultiSymmetry`` (``JOINT_ORDER_20`` plus a block
per sub-command), not the 12-DoF ``PiPlusSymmetry`` the other tasks use.

This config is standalone -- it deliberately does not derive from
``piplus_velocity_env_cfg`` -- so the two tasks can be tuned independently.
Shared settings (gait frequency, DR ranges, sensors, sim) are therefore
duplicated rather than inherited.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_piplus.objects import get_ball_cfg
from mjlab_piplus.robot.piplus_constants import (
  get_piplus_robot_cfg,
)
from mjlab_piplus.tasks import mdp as pi_mdp
from mjlab_piplus.tasks.mdp.footstep import (
  FootstepCommandCfg,
  footstep_tracking_roll_pitch,
  footstep_tracking_xy,
  footstep_tracking_yaw,
  footstep_tracking_z,
)
from mjlab_piplus.tasks.mdp.multi_rewards import MultiRewardTermCfg
from mjlab_piplus.tasks.mdp.path_following import (
  PathCommandCfg,
  align_with_path_reward,
  close_to_path_reward,
  goal_reached_angular,
  goal_reached_linear,
  path_progress_reward,
)

DELAY_MIN = 1
DELAY_MAX = 1

# Sensing delay of the ball observation, in control steps (200-300 ms at 50 Hz).
# The ball is seen by a camera rather than read off an encoder, so it lags far
# behind the proprioceptive terms above.
BALL_DELAY_MIN = 2
BALL_DELAY_MAX = 5
WALK_FREQUENCY = 1.75

# Top walking speed (m/s). One number, three places: the velocity command's
# ``lin_vel_x`` ceiling, and the progress caps of the path and goal-pose
# rewards. They are not literally the same quantity -- under a path or
# goal-pose command no twist is commanded at all, so there the cap bounds an
# *emergent* speed -- but they have to move together: a progress cap below the
# speed the robot can be asked for stops paying for speed it is still expected
# to make. Note the caps are a bare clamp, so raising this also raises those
# two terms' per-step head-room (at weight 2.0: 1.6 -> 3.0 at 0.8 -> 1.5),
# which shifts their share against the weight-1.0 goal/alignment terms.
# (The chris_dribble's ``max_speed`` is a different thing: closing speed on the ball.)
MAX_COMMANDED_SPEED = 0.8

# Angular tolerance (Gaussian sigma, radians) of the chris_dribble's signed aim reward,
# narrowed over training by the ``narrow_chris_dribble_aim`` curriculum. Both chris_dribble aim
# terms are signed: outside the tolerance they pay *negative*, so a tolerance
# that tightens faster than the policy can follow turns chris_dribbleing into a net loss
# and the robot stops attempting it. Hence a wide start and a long runway.
chris_dribble_DIR_SIGMA_START = math.radians(60.0)
chris_dribble_DIR_SIGMA_END = math.radians(20.0)

# Weight presets, spelled out so a reward's applicability reads at a glance.
_OFF = {
  "chris_dribble": 0.0,
}
chris_dribble_ONLY = {**_OFF, "chris_dribble": 1.0}

def chris_dribble_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  foot_sites = ("l_foot", "r_foot")
  # only for critic
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(
      ObjRef(type="site", name="l_foot", entity="robot"),
      ObjRef(type="site", name="r_foot", entity="robot"),
    ),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=False,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )

  ##
  # Observations
  ##
  actor_terms = {  # careful: when changing, also change symmetry config
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/gyro"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=NoiseModelWithAdditiveBiasCfg(
        noise_cfg=Unoise(n_min=-0.01, n_max=0.01),
        bias_noise_cfg=Unoise(n_min=-0.03, n_max=0.03, operation="add"),
        sample_bias_per_component=True,
      ),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      # biased=True for encoder-biased position
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "phase": ObservationTermCfg(
      func=pi_mdp.phase,
      params={"frequency": WALK_FREQUENCY, "command_name": "command"},
    ),
    # [one_hot(6) | velocity(3) | path(2N+2) | footstep(8M) | goal_pose(4)
    #  | chris_dribble(3)], inactive blocks zeroed.
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "command"},
    ),
    # Ball position and velocity in the base frame, zeroed off the chris_dribble
    # command. Its own term rather than part of the chris_dribble command block: it is
    # sensed, and on a camera's timescale -- the delay is applied inside the
    # term (see ``pi_mdp.ball_state``), not through delay_min_lag/delay_max_lag.
    "ball": ObservationTermCfg(
      func=pi_mdp.ball_state,
      params={
        "command_name": "command",
        "sub_command_name": "chris_dribble",
        "min_lag": BALL_DELAY_MIN,
        "max_lag": BALL_DELAY_MAX,
      },
    ),
  }
  critic_terms = {
    **actor_terms,
    # enable_corruption=False does not work for biased parameter
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": False},
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/local_linvel"},
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.1,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "command": pi_mdp.MultiCommandCfg(
      debug_vis=True,
      change_command_probability=0.3,
      fractions={
        "chris_dribble": 1.0,
      },
      terms={
        # Goal position for the ball; the ball itself is placed by the reset_ball event.
        "chris_dribble": pi_mdp.GoalPoseCommandCfg(
          entity_name="robot",
          debug_vis=True,
          resampling_time_range=(10.0, 25.0),
          radius_range=(1.0, 5.0)
        ),
      },
    )
  }

  ##
  # Events
  ##

  domain_rands = {
    # DR on startup, enough variance over 4096 envs, much better performance than per-step.
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          geom_names=(
            "l_ankle_roll_link_collision.*",
            "r_ankle_roll_link_collision.*",
          ),
        ),
        "operation": "scale",
        "ranges": (0.4, 1.0),
      },
    ),
    "joint_frictionloss": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "joint_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (1.0, 1.05),
        "operation": "scale",
      },
    ),
    "joint_damping": EventTermCfg(
      mode="startup",
      func=dr.joint_damping,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (0.95, 1.05),
        "operation": "scale",
      },
    ),
    "actuator_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(r".*",)),
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "link_masses": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "torso_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "ranges": (-0.25, 0.25),
        "operation": "add",
      },
    ),
    "link_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
        "operation": "add",
        "ranges": {
          0: (-0.01, 0.01),
          1: (-0.01, 0.01),
          2: (-0.01, 0.01),
        },
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        "operation": "add",
        "ranges": {
          0: (-0.03, 0.03),
          1: (-0.03, 0.03),
          2: (-0.03, 0.03),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "bias_range": (-0.1, 0.1),
      },
    ),
    "joint_torque": EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "effort_limit_range": (0.85, 1.0),
        "operation": "scale",
      },
    ),
  }
  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.0, 0.0),
          "y": (-0.0, 0.0),
          "z": (0.01, 0.01),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (-0.5, 0.5),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.25, 0.25),
          "y": (-0.25, 0.25),
          "z": (-0.15, 0.15),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
  } | domain_rands

  ##
  # Rewards
  ##

  # Path rewards evaluate the robot's position at the base footprint (the ground
  # midpoint between the two feet), so ``foot_asset_cfg`` carries the foot sites.
  foot_asset_cfg = SceneEntityCfg("robot", site_names=foot_sites)
  base_cfg = SceneEntityCfg("robot", body_names=("base_link",))
  # Foot *bodies* (not the sites above): the foot-placement penalties measure
  # link positions, and body_ids order follows the tuple below.
  feet_cfg = SceneEntityCfg(
    "robot", body_names=("r_ankle_roll_link", "l_ankle_roll_link")
  )

  # -- chris_dribble -----------------------------------------------------------------
  chris_dribble_rewards = {
    # One repeating cycle per ball: approach, strike, then a short window over
    # which the chris_dribble is scored densely before the ball is re-placed and the next
    # approach begins (see ``mdp/chris_dribble.py``). The window and the respawn are what
    # make chris_dribbleing worth more than loitering next to the ball -- the previous
    # design paid a single lump at contact and then nothing, so the approach was
    # worth more than the chris_dribble it was supposed to set up.
    #
    # Rough budget for one good chris_dribble, at these weights: contact +2.5 (until the
    # curriculum drops it), impact +6.0 over the 1 s window, direction +2.0 --
    # against roughly 1.5/s of approach reward on the way in. Keep that ordering
    # when retuning: Episode_Reward/chris_dribble_impact should dominate
    # Episode_Reward/chris_dribble_approach once the policy is chris_dribbleing at all.
    "chris_dribble_approach": MultiRewardTermCfg(
      func=pi_mdp.chris_dribble_approach,
      weight=3.0,
      params={
        "command_name": "command",
        "sub_command_name": "chris_dribble",
        # Speed toward the ball in m/s, capped: no bonus for charging it.
        "max_speed": 2.0,
        # Cubed cosine, so only a well-aimed approach gets most of the credit.
        "alignment_power": 3.0,
      },
      command_weights=chris_dribble_ONLY,
    ),
    "chris_dribble_face_ball": MultiRewardTermCfg(
      func=pi_mdp.chris_dribble_face_ball,
      weight=0.5,
      params={"command_name": "command", "sub_command_name": "chris_dribble"},
      command_weights=chris_dribble_ONLY,
    ),
    "chris_dribble_idle": MultiRewardTermCfg(
      func=pi_mdp.chris_dribble_idle,
      # Kills the do-nothing optimum: standing still with a ball to fetch has to
      # cost something, or refusing to engage is free.
      weight=-1.0,
      params={
        "command_name": "command",
        "sub_command_name": "chris_dribble",
        "vel_threshold": 0.1,
        # Inside this radius the robot is planting to swing, not loitering.
        "near_dist": 0.35,
      },
      command_weights=chris_dribble_ONLY,
    ),
    "chris_dribble_foot_alignment": MultiRewardTermCfg(
      func=pi_mdp.chris_dribble_foot_alignment,
      weight=-0,
      params={
        "command_name": "command",
        "sub_command_name": "chris_dribble",
        # Start turning the foot to line up well before the last half metre.
        "close_dist": 0.9,
      },
      command_weights=chris_dribble_ONLY,
    ),
    "chris_dribble_contact": MultiRewardTermCfg(
      func=pi_mdp.chris_dribble_contact,
      # Bootstrap only -- dropped to 0 by ``drop_chris_dribble_contact`` below. Left on,
      # a flat payout per touch makes a stream of weak taps competitive with
      # real chris_dribbles, since every touch buys a fresh ball.
      weight=0,
      params={
        "command_name": "command",
        "sub_command_name": "chris_dribble",
        "wrong_foot_penalty": 0.5,
      },
      command_weights=chris_dribble_ONLY,
    ),
    "chris_dribble_impact": MultiRewardTermCfg(
      func=pi_mdp.chris_dribble_impact,
      # The payload: paid every step of the window, so this is ~weight x
      # window_s for a perfect chris_dribble and as negative for a well-struck one sent
      # the wrong way.
      weight=0.5,
      params={
        "command_name": "command",
        "sub_command_name": "chris_dribble",
        # Aim tolerance in radians, narrowed by ``narrow_chris_dribble_aim`` below.
        "dir_sigma": chris_dribble_DIR_SIGMA_START,
        # Speed tolerance relative to the commanded strength, so 5 m/s is not
        # held to the same absolute precision as 1 m/s.
        "speed_rel_std": 0.35,
        "min_ball_speed": 0.2,
        # A toe-poke still pays a quarter of a clean side-foot contact: it has
        # to be discoverable before it can be refined away.
        "style_floor": 0.25,
      },
      command_weights=chris_dribble_ONLY,
    ),
    "chris_dribble_direction": MultiRewardTermCfg(
      func=pi_mdp.chris_dribble_direction,
      weight=2.0,
      params={
        "command_name": "command",
        "sub_command_name": "chris_dribble",
        "dir_sigma": chris_dribble_DIR_SIGMA_START,
        "min_ball_speed": 0.5,
      },
      command_weights=chris_dribble_ONLY,
    ),
  }

  # -- Stability ------------------------------------------------------------
  stability_rewards = {
    # Terminal penalty, paid once on the step "fell_over" fires. mjlab's
    # ``is_terminated`` excludes ``time_out=True`` terms, and the jittered
    # time-out is the only other termination, so this fires on falls and
    # nothing else. Scaled by dt like every weight here it costs 200 * 0.02 =
    # 4.0 once; the real cost of falling is the forfeited rest of the episode.
    #
    # (Every weight in this file is a per-step value: the reward manager scales
    # the lot by ``dt``, so the ratios between terms are what these numbers say.)
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=1.0,
      params={"std": math.sqrt(0.2), "asset_cfg": base_cfg},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.25,
      params={"asset_cfg": base_cfg},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.1,
      params={"sensor_name": "robot/root_angmom"},
    ),
  }

  # -- Safety ---------------------------------------------------------------
  safety_rewards = {
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-7,
    ),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
    ),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.05,
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={
        "sensor_name": "self_collision",
        "force_threshold": 2.0,
      },  # we dont want even soft collisions
    ),
    # Gated for getup
    "posture": MultiRewardTermCfg(
      func=pi_mdp.variable_posture,
      weight=1.0,
      command_weights=None,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "command",
        "std_standing": {
          r".*": 0.1,
        },
        "std_walking": {
          r".*_hip_pitch_joint": 0.35,
          r".*_hip_roll_joint": 0.15,
          r".*_thigh_joint": 0.15,
          r".*_calf_joint": 0.35,
          r".*_ankle_pitch_joint": 0.35,
          r".*_ankle_roll_joint": 0.1,
          r".*_shoulder_pitch_joint": 0.25,
          r".*_shoulder_roll_joint": 0.1,
          r".*_upper_arm_joint": 0.05,
          r".*_elbow_joint": 0.2,
        },
        "std_running": {
          r".*_hip_pitch_joint": 0.5,
          r".*_hip_roll_joint": 0.25,
          r".*_thigh_joint": 0.25,
          r".*_calf_joint": 0.5,
          r".*_ankle_pitch_joint": 0.5,
          r".*_ankle_roll_joint": 0.1,
          r".*_shoulder_pitch_joint": 0.35,
          r".*_shoulder_roll_joint": 0.15,
          r".*_upper_arm_joint": 0.05,
          r".*_elbow_joint": 0.25,
        },
        # NOTE: under a path command the commanded twist reads zero, so this
        # term sees the walking regime; only a velocity command can select
        # ``std_running``.
        "running_threshold": 0.5,
      },
    ),
  }

  # -- Gait shaping ---------------------------------------------------------
  gait_rewards = {
    # Encourage feet to be at target height when the foot velocity is high
    # (i.e. during swing).
    # The three gait terms are gated on being upright as well: face down, the
    # gait clock keeps running and would pay for chris_dribbleing the floor on the beat,
    # and the slip penalty would charge for exactly the ground contact the arms
    # and feet need to push off with. All three come back as the robot rises.
    "foot_gait": RewardTermCfg(
      func=pi_mdp.feet_gait,
      weight=1.0,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_name": "command",
        "sensor_name": "feet_ground_contact",
      },
    ),
    "feet_phase": RewardTermCfg(
      func=pi_mdp.feet_phase,
      weight=1.0,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": [0.0, 0.5],
        "foot_height": 0.03,
        "stance_threshold": 0.56,
        "std": 0.025,
        "command_name": "command",
        "asset_cfg": SceneEntityCfg("robot"),
        # Clearance is the lowest collision geom per foot (tilt-invariant).
        # Order matches `offset`: right (0.0) first, left (0.5) second.
        "foot_geom_names": [
          ("r_ankle_roll_link_collision.*",),
          ("l_ankle_roll_link_collision.*",),
        ],
        # Resting clearance of the lowest geom center when the foot is
        # planted; subtracted so a planted foot reads ~0. The min is over the
        # four sole capsules (radius 0.01), so a foot flat on the plane reads
        # ~0.0098 -- measured by settling the robot on a plane with ctrl held
        # at the home pose. NOT 0.019: that is the value at the home keyframe,
        # where the robot floats 9.1 mm above the ground.
        "baseline_height": 0.0098,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=pi_mdp.feet_slip,
      # Ramped in by the curriculum; see "slip_weight".
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "command",
        "asset_cfg": foot_asset_cfg,
      },
    ),
  }

  rewards = (
    chris_dribble_rewards
    | stability_rewards
    | safety_rewards
    | gait_rewards
  )

  ##
  # Metrics
  ##
  metrics = {
    "feet_swing_peak_height": MetricsTermCfg(
      func=pi_mdp.feet_swing_peak_height,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": foot_asset_cfg,
      },
    ),
  }

  ##
  # Terminations
  ##
  terminations = {
    # Jittered per-env timeout keeps episode resets decorrelated
    "time_out": TerminationTermCfg(
      func=pi_mdp.time_out_jittered,
      time_out=True,
      params={"min_fraction": 0.5},
    ),
    # A fall ends the episode. This was a getup *deadline* while the robot was
    # expected to recover from falls; with that gone, the walking convention
    # applies again -- the same 70 deg limit the velocity task uses.
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={
        "limit_angle": math.radians(60.0)
      },  # 0 deg upright, 90 deg would be flat on the ground
    ),
    # Backstop only. A squat is perfectly vertical, so `fell_over` never sees
    # it; this is the one term that does.
    "root_height_below_minimum": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.25},
    ),
  }

  ##
  # Curriculum
  ##
  curriculum = {
    # chris_dribble bootstrap off: the flat payout per contact exists only to get the
    # policy to touch the ball at all. Once it does, the quality terms should
    # drive the rest, or weak taps stay competitive with real chris_dribbles.
    "drop_chris_dribble_contact": CurriculumTermCfg(
      func=pi_mdp.reward_weight,
      params={
        "reward_name": "chris_dribble_contact",
        "weight_stages": [{"step": 1500 * 24, "weight": 0.0}],
      },
    ),
    # Then tighten the aim tolerance, on both aim terms at once so they keep
    # scoring the same chris_dribble the same way.
    "narrow_chris_dribble_aim": CurriculumTermCfg(
      func=pi_mdp.reward_param_linear,
      params={
        "reward_names": ["chris_dribble_impact", "chris_dribble_direction"],
        "param_name": "dir_sigma",
        "start_step": 1500 * 24,
        "end_step": 6400 * 24,
        "start_value": chris_dribble_DIR_SIGMA_START,
        "end_value": chris_dribble_DIR_SIGMA_END,
      },
    ),
  }

  ##
  # Sensors
  ##

  feet_contact_sensor_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(l_ankle_roll_link|r_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=4,
  )
  # Fires only when the camera bar itself touches the ground -- see the
  # "camera_ground_impact" reward. Geom-mode, not subtree: the head shell is
  # allowed to take a landing, the lens is not.
  camera_ground_contact_cfg = ContactSensorCfg(
    name="camera_ground_contact",
    primary=ContactMatch(
      mode="geom", pattern=r"^camera_link_collision[0-9]$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  # Matches the *_selfpad geoms, not the real collision geoms: the pads sit on
  # their own contype/conaffinity bit, so they only ever touch each other --
  # never the terrain, never the ball. Parent-child pairs are dropped by MuJoCo's
  # filterparent, so adjacent pads (thigh<->calf, base<->torso) never fire.
  #
  # NOTE: no ``secondary``. A secondary *pattern* would silently collapse to a
  # single geom -- ``secondary_policy`` defaults to "first" -- so matching
  # ".*_selfpad" on both sides scores every pad against the base pad alone and
  # reads ~0 no matter how folded the robot is. It is not needed: the bit-2
  # isolation already guarantees a pad's only possible contact partner is
  # another pad, so "any contact involving this pad" *is* pad-vs-pad.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="geom", pattern=r".*_selfpad", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  scene = SceneCfg(
    num_envs=4096,
    env_spacing=0.0,
    entities={
      # 20 DoF: the arms are actuated here, unlike in the velocity and path
      # tasks. A robot that cannot use its arms cannot roll over and push
      # itself up, and they are worth having for balance while walking anyway.
      # self_pads activates the *_selfpad geoms that the self_collision sensor
      # scores -- see SELF_PAD_COLLISION.
      "robot": get_piplus_robot_cfg(with_arms=True, self_pads=True),
      "ball": get_ball_cfg(),
    },
    sensors=(
      feet_contact_sensor_cfg,
      self_collision_cfg,
      camera_ground_contact_cfg,
      foot_height_scan,
    ),
    terrain=TerrainEntityCfg(
      terrain_type="plane",
      terrain_generator=None,
      max_init_terrain_level=0,
      env_spacing=0.0,
    ),
  )

  ##
  # Viewer
  ##
  viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="base_link",
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
    # This task has 34 reward terms; the default of 20 silently drops the rest
    # from the viser reward bar panel.
    reward_bar_max_terms=40,
  )

  ##
  # Simulation
  ##
  sim = SimulationCfg(
    # Sized from measurement, not from the getup-era 120 (a robot lying on its
    # back touches the floor with the whole trunk, both arms and both legs at
    # once, ~43 contacts). With falls terminal the robot never reaches those
    # poses, and the contact count is bounded by how many collision geoms can
    # touch a plane at once: measured peak is 19 in any single world across
    # 512 envs x 400 steps of standing, of thrashing under random actions, and
    # of being force-tipped to 65 and 89 degrees. 30 is ~58% headroom.
    #
    # Contacts past nconmax are dropped *silently* and look like the robot
    # sinking through the floor, so re-measure before trimming further -- and
    # raise it again for anything that puts the robot on the ground on purpose,
    # or for terrain that is not a plane.
    nconmax=45,
    njmax=200,
    # contact_sensor_maxmatch=500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
      cone="pyramidal",
      # ccd_iterations=500,
    ),
  )
  episode_length_s = 20.0

  if play:
    scene.num_envs = 1
    events = {  # remove other dr in play mode
      "reset_base": events["reset_base"],
      "reset_robot_joints": events["reset_robot_joints"],
    }
    # Unbounded episodes in play: plain time_out (never fires at 1e10 s).
    terminations["time_out"] = TerminationTermCfg(func=mdp.time_out, time_out=True)
    episode_length_s = 1e10
    curriculum = {}
    # With no episode resets in play, the type only changes on the active
    # sub-command's own timer (up to 30 s for a path). Use the viewer's
    # command-type dropdown to switch on demand.
    # Keep the play library small enough to generate in a few seconds, but not
    # so small that every episode shows the same worlds.
    metrics["phase_metric"] = MetricsTermCfg(
      func=pi_mdp.phase_metric,
      params={"frequency": WALK_FREQUENCY, "command_name": "command"},
    )
    # chris_dribble diagnostics: how fast the ball is actually going, and where in the
    # ball cycle the robot is (0 = not chris_dribbleing, 1 = approaching, 2 = inside the
    # post-chris_dribble scoring window). Play only -- in training these would just be
    # averaged over thousands of envs in different phases. The per-chris_dribble numbers
    # (peak speed, aim error, style) are logged by the command itself and show
    # up in training under Metrics/chris_dribble_*.
    metrics["chris_dribble_ball_speed"] = MetricsTermCfg(
      func=pi_mdp.chris_dribble_ball_speed,
      params={"command_name": "command", "sub_command_name": "chris_dribble"},
    )
    metrics["chris_dribble_phase"] = MetricsTermCfg(
      func=pi_mdp.chris_dribble_phase,
      params={"command_name": "command", "sub_command_name": "chris_dribble"},
      # 0/1/2 is a state, not a quantity: averaging it over an episode says
      # nothing, so report where the episode ended up.
      reduce="last",
    )
    metrics["foot_air_time"] = MetricsTermCfg(
      func=pi_mdp.current_air_time,
      params={"sensor_name": "feet_ground_contact"},
    )
    metrics["last_air_time"] = MetricsTermCfg(
      func=pi_mdp.last_air_time,
      params={"sensor_name": "feet_ground_contact"},
    )
    for side in ("l", "r"):
      for axis in ("roll", "pitch"):
        metrics[f"{side}_foot_{axis}"] = MetricsTermCfg(
          func=pi_mdp.foot_tilt,
          params={
            "asset_cfg": SceneEntityCfg("robot", site_names=(f"{side}_foot",)),
            "axis": axis,
          },
        )
    # NOTE: SceneEntityCfg.resolve() sorts sites by model index, so the reward's
    # site order is (r_foot=idx0, l_foot=idx1) -- i.e. offset[0]=0.0 drives the
    # RIGHT foot and offset[1]=0.5 the LEFT foot. The metric offsets below are
    # matched to that so each label tracks the correct physical foot.
    metrics["l_foot_rz_target"] = MetricsTermCfg(
      func=pi_mdp.feet_phase_target,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": 0.5,
        "foot_height": 0.045,
        "stance_threshold": 0.56,
      },
    )
    metrics["r_foot_rz_target"] = MetricsTermCfg(
      func=pi_mdp.feet_phase_target,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": 0.0,
        "foot_height": 0.045,
        "stance_threshold": 0.56,
      },
    )

  ##
  # Assemble the config
  ##
  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    metrics=metrics,
    terminations=terminations,
    curriculum=curriculum,
    viewer=viewer,
    sim=sim,
    decimation=4,
    episode_length_s=episode_length_s,
  )
