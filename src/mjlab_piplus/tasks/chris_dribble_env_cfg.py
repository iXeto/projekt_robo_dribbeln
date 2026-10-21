"""Dribble task: walk with the ball at the feet.

Standalone copy of ``piplus_velocity_env_cfg``, not derived from it, so the
walking settings can be tuned for dribbling without touching the velocity task.
The ball parts are added on top of this copy step by step.
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
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, TerrainHeightSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg
from mjlab_piplus.tasks import mdp as pi_mdp
from mjlab_piplus.terrain import AstroturfTerrainCfg

DELAY_MIN = 1
DELAY_MAX = 1
WALK_FREQUENCY = 1.75


def chris_dribble_env_cfg(
  play: bool = False, static: bool = False
) -> ManagerBasedRlEnvCfg:
  foot_sites = ("l_foot", "r_foot")
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
        noise_cfg=Unoise(
          n_min=-0.01,
          n_max=0.01,
        ),
        bias_noise_cfg=Unoise(n_min=-0.03, n_max=0.03, operation="add"),
        sample_bias_per_component=True,
      ),
      delay_min_lag=DELAY_MIN,
      delay_max_lag=DELAY_MAX,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      # biased=True so the actor reads the encoder-biased position (true + bias),
      # matching the action manager which always compensates the target by
      # -encoder_bias. Without this the encoder_bias DR is only half-applied (the
      # actor sees the true position) and models no real calibration error.
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
    "actions": ObservationTermCfg(
      func=mdp.last_action,
      # noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "phase": ObservationTermCfg(
      func=pi_mdp.phase,
      params={
        "frequency": WALK_FREQUENCY,
        "command_name": "twist",
      },
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }
  critic_terms = {
    **actor_terms,
    # Critic is privileged: override the (biased) actor joint_pos with the true
    # position so the value function sees ground truth.
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
    # "foot_height": ObservationTermCfg(
    #   func=mdp.foot_height,
    #   params={"sensor_name": "foot_height_scan"},
    # ),
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
    "twist": pi_mdp.UniformVelocityCommandWithStopCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      debug_vis=True,
      heading_command=True,
      heading_control_stiffness=0.0,
      rel_standing_envs=0.075,
      rel_walk_in_place_envs=0.075,
      rel_heading_envs=0.3,
      rel_forward_envs=0.15,
      ranges=pi_mdp.UniformVelocityCommandWithStopCfg.Ranges(
        lin_vel_x=(-0.5, 1.0),
        lin_vel_y=(-0.25, 0.25),
        ang_vel_z=(-1.5, 1.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

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
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=("l_ankle_roll_link_collision.*", "r_ankle_roll_link_collision.*"),),
        "operation": "scale",
        "ranges": (0.4, 1.0),
      }
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
    # NOTE: mass/COM/armature DR runs in "startup" mode, not "reset". Each of
    # these triggers a MuJoCo constant recompute (body_mass/body_ipos ->
    # set_const, dof_armature -> set_const_0) that is NOT in the CUDA graph.
    # In "reset" mode this recompute fired on nearly every step (envs reset
    # constantly early in training), consuming ~21% of total CPU and capping
    # throughput. Running them once per env at startup keeps per-env diversity
    # across the 4096 worlds while paying the recompute a single time.
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

  ##
  # Rewards
  ##
  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.125)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    # stability rewards
    "upright": RewardTermCfg(
        func=mdp.upright,
        weight=1.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
        },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.25,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.1,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    # safety rewards
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={
        "sensor_name": "self_collision",
        "force_threshold": 2.0,
      },  # we dont want even soft collisions
    ),
    "pose": RewardTermCfg(
      func=pi_mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {r".*": 0.05,},
        "std_walking": {
          r".*_hip_pitch_joint": 0.35,
          r".*_hip_roll_joint": 0.15,
          r".*_thigh_joint": 0.15,
          r".*_calf_joint": 0.35,
          r".*_ankle_pitch_joint": 0.35,
          r".*_ankle_roll_joint": 0.1,
        },
        "std_running": {
          r".*_hip_pitch_joint": 0.5,
          r".*_hip_roll_joint": 0.25,
          r".*_thigh_joint": 0.25,
          r".*_calf_joint": 0.5,
          r".*_ankle_pitch_joint": 0.5,
          r".*_ankle_roll_joint": 0.1,
        },
        "running_threshold": 0.5,
      },
    ),
    # shaping rewards
    # encourage feet to be at target height when the foot velocity is high (i.e. during swing)
    "foot_gait": RewardTermCfg(
      func=pi_mdp.feet_gait,
      weight=1.0,
      params={
        "frequency": WALK_FREQUENCY,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_name": "twist",
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
        "command_name": "twist",
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
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
      },
    ),
    # "soft_landing": RewardTermCfg(
    #   func=pi_mdp.soft_landing,
    #   weight=-1e-3,
    #   params={
    #     "sensor_name": "feet_ground_contact",
    #     "command_name": "twist",
    #   },
    # ),
    # "flat_foot": RewardTermCfg(
    #   func=pi_mdp.flat_foot_l2,
    #   weight=-0.0,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
    #   },
    # ),
  }


  ##
  # Metrics
  ##
  metrics = {
    #"base_ang_vel_xy_l2": MetricsTermCfg(
    #  func=pi_mdp.base_ang_vel_xy_l2,
    #),
    #"step_frequency": MetricsTermCfg(
    #  func=pi_mdp.step_frequency,
    #  params={"sensor_name": "feet_ground_contact"},
    #),
    #"foot_contact_force": MetricsTermCfg(
    #  func=pi_mdp.foot_contact_force,
    #  params={"sensor_name": "feet_ground_contact"},
    #),
    "feet_swing_peak_height": MetricsTermCfg(
      func=pi_mdp.feet_swing_peak_height,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites),
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
    #"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    # "root_height_below_minimum": TerminationTermCfg(
    #  func=mdp.root_height_below_minimum,
    #  params={"minimum_height": 0.25},
    # ),
  }

  ##
  # Curriculum
  ##
  curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {
            "step": 0 * 24,
            "lin_vel_x": (-0.25, 0.35),
            "lin_vel_y": (-0.15, 0.15),
            "ang_vel_z": (-math.pi / 4, math.pi / 4),
          },
          {
            "step": 800 * 24,
            "lin_vel_x": (-0.5, 0.8),
            "lin_vel_y": (-0.4, 0.4),
            "ang_vel_z": (-1.5, 1.5),
          },
        ],
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
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  scene = SceneCfg(
    num_envs=4096,
    env_spacing=0.0,
    entities={"robot": get_piplus_robot_cfg()},
    sensors=(feet_contact_sensor_cfg, self_collision_cfg,), # foot_height_scan),
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
  )

  ##
  # Simulation
  ##
  sim = SimulationCfg(
    nconmax=45,
    njmax=1500,
    contact_sensor_maxmatch=500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
      ccd_iterations=500,
    ),
  )
  episode_length_s = 20.0
  if play:
    metrics["phase_metric"] = MetricsTermCfg(
      func=pi_mdp.phase_metric,
      params={
        "frequency": WALK_FREQUENCY,
        "command_name": "twist",
      },
    )
    scene.num_envs = 1
    events = {  # remove other dr in play mode
      "reset_base": events["reset_base"],
      "reset_robot_joints": events["reset_robot_joints"],
      # "push_robot": events["push_robot"],
    }
    # Unbounded episodes in play: plain time_out (never fires at 1e10 s).
    terminations["time_out"] = TerminationTermCfg(func=mdp.time_out, time_out=True)
    episode_length_s = 1e10
    curriculum = {}
    metrics["foot_air_time"] = MetricsTermCfg(
      func=pi_mdp.current_air_time,
      params={"sensor_name": "feet_ground_contact"},
    )
    metrics["last_air_time"] = MetricsTermCfg(
      func=pi_mdp.last_air_time,
      params={"sensor_name": "feet_ground_contact"},
    )
    metrics["l_foot_roll"] = MetricsTermCfg(
      func=pi_mdp.foot_tilt,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=("l_foot",)),
        "axis": "roll",
      },
    )
    metrics["l_foot_pitch"] = MetricsTermCfg(
      func=pi_mdp.foot_tilt,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=("l_foot",)),
        "axis": "pitch",
      },
    )
    metrics["r_foot_roll"] = MetricsTermCfg(
      func=pi_mdp.foot_tilt,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=("r_foot",)),
        "axis": "roll",
      },
    )
    metrics["r_foot_pitch"] = MetricsTermCfg(
      func=pi_mdp.foot_tilt,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=("r_foot",)),
        "axis": "pitch",
      },
    )
    # NOTE: SceneEntityCfg.resolve() sorts sites by model index, so the reward's
    # site order is (r_foot=idx0, l_foot=idx1) — i.e. offset[0]=0.0 drives the
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
