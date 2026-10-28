"""Chris dribble env: dribble the ball to a goal position.

Started as a copy of the multi task (DR, sensor noise, sensors, sim, stability,
safety and gait terms are still its own), with one command: a bare
``GoalPoseCommand`` named ``chris_dribble`` -- no ``MultiCommand``. Its goal is
where the *ball* has to end up; the policy sees only the goal's xy in the
robot's yaw frame. ``stand_mask`` / ``velocity_command`` answer "never" and
"zero" for a bare goal command, which keeps the gait, phase and posture terms
in the walking regime.

The ball is placed just in front of the robot at every reset and observed with
a camera's delay. The rewards are the mini dribble task's: keep the ball in
front of the feet, move it towards the goal, settle it on the goal, face the
goal. Losing the ball ends the episode.

This config is standalone -- it deliberately does not derive from
``piplus_velocity_env_cfg`` -- so the tasks can be tuned independently.
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

DELAY_MIN = 1
DELAY_MAX = 1

# Sensing delay of the ball observation, in control steps (40-100 ms at 50 Hz).
# The ball is seen by a camera rather than read off an encoder, so it lags far
# behind the proprioceptive terms above.
BALL_DELAY_MIN = 2
BALL_DELAY_MAX = 5
WALK_FREQUENCY = 1.75

COMMAND_NAME = "chris_dribble"
# Ball speed (m/s) above which moving the ball towards the goal earns nothing
# more.
MAX_BALL_SPEED = 0.4
BALL_LOST_DISTANCE = 1.0


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
      params={"frequency": WALK_FREQUENCY, "command_name": COMMAND_NAME},
    ),
    # Ball goal xy relative to the base, yaw frame.
    "command": ObservationTermCfg(
      func=pi_mdp.goal_pos_b,
      params={"command_name": COMMAND_NAME},
    ),
    # Ball position and velocity in the base frame. Sensed by a camera, so it
    # lags far behind the proprioceptive terms.
    "ball": ObservationTermCfg(
      func=pi_mdp.relative_ball_position,
      params={"ball_entity_name": "ball"},
      delay_min_lag=BALL_DELAY_MIN,
      delay_max_lag=BALL_DELAY_MAX,
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
    # Privileged: the true ball state, without the camera delay.
    "ball": ObservationTermCfg(
      func=pi_mdp.relative_ball_position,
      params={"ball_entity_name": "ball"},
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

  # Where the ball has to go, sampled relative to the robot.
  commands: dict[str, CommandTermCfg] = {
    COMMAND_NAME: pi_mdp.GoalPoseCommandCfg(
      entity_name="robot",
      debug_vis=True,
      resampling_time_range=(10.0, 25.0),
      radius_range=(1.0, 5.0),
    ),
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
          # Fixed so that reset_ball's spawn point, which is in the world
          # frame, lands in front of the robot.
          "yaw": (0.0, 0.0),
        },
        "velocity_range": {},
      },
    ),
    # Ball 0.2-0.35 m ahead of the robot, at rest.
    "reset_ball": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (0.2, 0.35), "y": (-0.05, 0.05)},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("ball"),
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

  foot_asset_cfg = SceneEntityCfg("robot", site_names=foot_sites)
  base_cfg = SceneEntityCfg("robot", body_names=("base_link",))

  # -- Dribble ------------------------------------------------------------
  # Progress carries the ball to the goal, ball_at_goal settles it there, and
  # the look-at term keeps the robot behind the ball on the line to the goal.
  # Weights are the mini dribble task's.
  goal_params = {"command_name": COMMAND_NAME}
  dribble_rewards = {
    "ball_in_front": RewardTermCfg(
      func=pi_mdp.ball_in_front,
      weight=1.5,
      params={"std": 0.3, "target_offset": 0.25},
    ),
    "ball_goal_progress": RewardTermCfg(
      func=pi_mdp.ball_goal_progress,
      weight=4.0,
      params={**goal_params, "max_speed": MAX_BALL_SPEED},
    ),
    "ball_at_goal": RewardTermCfg(
      func=pi_mdp.ball_at_goal,
      weight=2.0,
      params={**goal_params, "std": 0.3},
    ),
    "look_at_goal": RewardTermCfg(
      func=pi_mdp.orient_towards_goal,
      weight=0.5,
      params={**goal_params, "coeff": 1.0, "distance_threshold": 0.3},
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
    "posture": RewardTermCfg(
      func=pi_mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": COMMAND_NAME,
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
        # NOTE: a goal command reads as a zero twist, so this term always sees
        # the walking regime; ``std_running`` is never selected here.
        "running_threshold": 0.5,
      },
    ),
  }

  # -- Gait shaping ---------------------------------------------------------
  gait_rewards = {
    # Encourage feet to be at target height when the foot velocity is high
    # (i.e. during swing).
    "foot_gait": RewardTermCfg(
      func=pi_mdp.feet_gait,
      weight=1.0,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_name": COMMAND_NAME,
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
        "command_name": COMMAND_NAME,
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
        "command_name": COMMAND_NAME,
        "asset_cfg": foot_asset_cfg,
      },
    ),
  }

  rewards = dribble_rewards | stability_rewards | safety_rewards | gait_rewards

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
    "ball_lost": TerminationTermCfg(
      func=pi_mdp.ball_too_far,
      params={"max_distance": BALL_LOST_DISTANCE},
    ),
  }

  ##
  # Curriculum
  ##
  curriculum: dict[str, CurriculumTermCfg] = {}

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
    metrics["phase_metric"] = MetricsTermCfg(
      func=pi_mdp.phase_metric,
      params={"frequency": WALK_FREQUENCY, "command_name": COMMAND_NAME},
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
