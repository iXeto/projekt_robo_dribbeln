"""Melika dribble task: copy of the mini dribble task, to be hardened for sim2real.

Built on top of the PiPlus velocity task from mjlab_piplus, which already walks.
Added on top of it:

* actuated arms (20 DoF, like the multi task) instead of the 12-DoF legs only;
* a ball (RoboCup kid-size, from ``mjlab_piplus.objects``), placed just in front
  of the robot at every reset;
* the ball's position and velocity in the robot's yaw frame: noisy and on a
  camera's 2-5 step delay for the actor, true state for the critic;
* a goal position instead of the walking velocity command: the ball has to be
  dribbled there. The policy sees only the goal's xy in the robot's yaw frame,
  the same thing localization would hand it on the real robot;
* rewards for keeping the ball in front of the feet, moving the ball towards
  the goal, the ball sitting on the goal, and facing the goal;
* a termination when the ball is more than ``BALL_LOST_DISTANCE`` away.

The goal is the multi task's ``GoalPoseCommand``, wrapped in a ``MultiCommand``
with that single type. The wrapper is plumbing only: the gait, phase and
posture terms ask the command whether the robot should stand (``stand_mask``)
and how fast it is commanded to go (``velocity_command``). A ``MultiCommand``
without a stand or velocity type answers "never" and "zero", which keeps them
in the walking regime; asked directly, a bare goal command would have its
``sin(yaw_err)`` read as the stand flag.

The robot always spawns facing +x so that the ball's spawn point is in front of
it, and the goal is placed inside a cone ahead of the robot.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_piplus.objects import get_ball_cfg
from mjlab_piplus.robot.piplus_constants import get_piplus_robot_cfg
from mjlab_piplus.tasks import mdp
from mjlab_piplus.tasks.piplus_velocity_env_cfg import piplus_velocity_env_cfg

BALL_LOST_DISTANCE = 1.0
EPISODE_LENGTH_S = 15.0
# Ball speed (m/s) above which moving the ball towards the goal earns nothing
# more. The velocity task's slow dribbling range topped out here too.
MAX_BALL_SPEED = 0.4
COMMAND_NAME = "command"
GOAL_NAME = "goal"

# Sensing delay of the ball, in control steps (40-100 ms at 50 Hz). The ball is
# seen by a camera rather than read off an encoder, so it lags far behind the
# proprioceptive terms, which carry one step. Same values as the multi task.
BALL_DELAY_MIN = 2
BALL_DELAY_MAX = 5
# Uniform noise on the actor's ball observation, per component of
# ``[x, y, vx, vy]``: m for the position, m/s for the velocity. The velocity is
# a difference of noisy camera positions, hence the larger band. Guesses, not
# measured on the robot -- replace them once the vision pipeline is profiled.
BALL_POS_NOISE = 0.05
BALL_VEL_NOISE = 0.2
# The goal comes from localization, which runs on the same camera frames.
GOAL_DELAY_MIN = 2
GOAL_DELAY_MAX = 5
GOAL_POS_NOISE = 0.1


def melika_dribble_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = piplus_velocity_env_cfg(play=play)

  ##
  # Scene: add the ball. The robot plus ball touch more than the walking task's
  # contact budget, and dropped contacts look like sinking through the floor.
  ##
  cfg.scene.entities["ball"] = get_ball_cfg()
  cfg.sim.nconmax = 60

  ##
  # Robot: actuate the arms (20 DoF instead of 12), as the multi task does, for
  # balance while handling the ball. Changes the action and joint obs widths.
  ##
  # self_pads activates the *_selfpad geoms, which the self_collision sensor
  # below scores instead of the velocity task's base_link subtree match.
  cfg.scene.entities["robot"] = get_piplus_robot_cfg(with_arms=True, self_pads=True)
  # The multi task's pad sensor. NOTE: no ``secondary``: a secondary pattern
  # collapses to one geom (``secondary_policy`` defaults to "first") and would
  # score every pad against the base pad alone. The pads sit on their own
  # contype/conaffinity bit, so any contact a pad has is with another pad.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="geom", pattern=r".*_selfpad", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = tuple(
    self_collision_cfg if s.name == "self_collision" else s for s in cfg.scene.sensors
  )
  # The velocity task's posture stds only name the leg joints. Arm values are
  # the multi task's.
  pose_params = cfg.rewards["pose"].params
  pose_params["std_walking"] = {
    **pose_params["std_walking"],
    r".*_shoulder_pitch_joint": 0.25,
    r".*_shoulder_roll_joint": 0.1,
    r".*_upper_arm_joint": 0.05,
    r".*_elbow_joint": 0.2,
  }
  pose_params["std_running"] = {
    **pose_params["std_running"],
    r".*_shoulder_pitch_joint": 0.35,
    r".*_shoulder_roll_joint": 0.15,
    r".*_upper_arm_joint": 0.05,
    r".*_elbow_joint": 0.25,
  }

  ##
  # Command: a goal position for the ball, replacing the walking twist. One goal
  # per episode, 1-3 m away, within 45 degrees of where the robot faces.
  ##
  cfg.commands = {
    COMMAND_NAME: mdp.MultiCommandCfg(
      resampling_time_range=(0.0, 0.0),
      # The wrapper's own flag gates its sub-terms' drawing too.
      debug_vis=True,
      terms={
        GOAL_NAME: mdp.GoalPoseCommandCfg(
          entity_name="robot",
          debug_vis=True,
          resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
          radius_range=(1.0, 5.0),
          bearing_range=(-math.pi / 4, math.pi / 4),
        ),
      },
      fractions={GOAL_NAME: 1.0},
      change_command_probability=0.0,
    )
  }
  # Every term that read the twist now reads the wrapper.
  for group in cfg.observations.values():
    for term in group.terms.values():
      if term.params.get("command_name") == "twist":
        term.params["command_name"] = COMMAND_NAME
  for term in [*cfg.rewards.values(), *cfg.metrics.values()]:
    if term.params.get("command_name") == "twist":
      term.params["command_name"] = COMMAND_NAME
  # Nothing commands a twist any more: the ball's progress sets the pace.
  cfg.rewards.pop("track_linear_velocity")
  cfg.rewards.pop("track_angular_velocity")
  # The velocity curriculum widens the twist ranges, which are gone.
  cfg.curriculum = {}

  ##
  # Observations: ball state and goal for actor and critic. The goal takes the
  # twist's place in the observation. The actor sees both the way the robot
  # will: noisy and on a camera's delay. The critic sees the true state.
  ##
  goal_params = {"command_name": COMMAND_NAME, "sub_command_name": GOAL_NAME}
  cfg.observations["actor"].terms.update(
    {
      "command": ObservationTermCfg(
        func=mdp.goal_pos_b,
        params=goal_params,
        noise=Unoise(n_min=-GOAL_POS_NOISE, n_max=GOAL_POS_NOISE),
        delay_min_lag=GOAL_DELAY_MIN,
        delay_max_lag=GOAL_DELAY_MAX,
      ),
      # Position and velocity in one term, so both lag by the same frame.
      "ball": ObservationTermCfg(
        func=mdp.ball_state_b,
        noise=Unoise(
          n_min=(-BALL_POS_NOISE,) * 2 + (-BALL_VEL_NOISE,) * 2,
          n_max=(BALL_POS_NOISE,) * 2 + (BALL_VEL_NOISE,) * 2,
        ),
        delay_min_lag=BALL_DELAY_MIN,
        delay_max_lag=BALL_DELAY_MAX,
      ),
    }
  )
  cfg.observations["critic"].terms.update(
    {
      "command": ObservationTermCfg(func=mdp.goal_pos_b, params=goal_params),
      "ball_pos": ObservationTermCfg(func=mdp.ball_pos_b),
      "ball_vel": ObservationTermCfg(func=mdp.ball_vel_b),
    }
  )

  ##
  # Events: robot faces +x, ball spawns 0.2-0.35 m ahead of it. The velocity
  # task's push_robot stays on: being bumped while on the ball is the normal
  # case on a real pitch.
  ##
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
  cfg.events["reset_ball"] = EventTermCfg(
    func=vel_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"x": (0.2, 0.35), "y": (-0.05, 0.05)},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("ball"),
    },
  )

  ##
  # Rewards: keep the walking terms, add the ball and the goal. Progress carries
  # the ball there, ball_at_goal settles it on the goal, and the look-at term
  # keeps the robot behind the ball on the line to the goal.
  ##
  cfg.rewards["ball_in_front"] = RewardTermCfg(
    func=mdp.ball_in_front,
    weight=1.5,
    params={"std": 0.3, "target_offset": 0.25},
  )
  cfg.rewards["ball_goal_progress"] = RewardTermCfg(
    func=mdp.ball_goal_progress,
    weight=4.0,
    params={**goal_params, "max_speed": MAX_BALL_SPEED},
  )
  cfg.rewards["ball_at_goal"] = RewardTermCfg(
    func=mdp.ball_at_goal,
    weight=2.0,
    params={**goal_params, "std": 0.3},
  )
  cfg.rewards["look_at_goal"] = RewardTermCfg(
    func=mdp.orient_towards_goal,
    weight=0.5,
    params={**goal_params, "coeff": 1.0, "distance_threshold": 0.3},
  )

  ##
  # Terminations: the multi task's fall limits, and ball lost.
  ##
  # 60 deg instead of the velocity task's 70: the real robot is past saving
  # well before 70.
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(60.0)
  # Backstop: a squat is perfectly vertical, so fell_over never sees it.
  cfg.terminations["root_height_below_minimum"] = TerminationTermCfg(
    func=vel_mdp.root_height_below_minimum,
    params={"minimum_height": 0.25},
  )
  cfg.terminations["ball_lost"] = TerminationTermCfg(
    func=mdp.ball_too_far,
    params={"max_distance": BALL_LOST_DISTANCE},
  )

  if not play:
    cfg.episode_length_s = EPISODE_LENGTH_S

  return cfg
