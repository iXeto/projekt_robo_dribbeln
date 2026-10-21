"""Mini dribble task: a smoke test, not a tuned policy.

Built on top of the PiPlus velocity task from mjlab_piplus, which already walks.
Added on top of it:

* a ball (RoboCup kid-size, from ``mjlab_piplus.objects``), placed just in front
  of the robot at every reset;
* the ball's position and velocity in the robot's yaw frame, for actor and
  critic (true state: no noise, no delay);
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
from mjlab.tasks.velocity import mdp as vel_mdp

from mjlab_piplus.objects import get_ball_cfg
from mjlab_piplus.tasks import mdp
from mjlab_piplus.tasks.piplus_velocity_env_cfg import piplus_velocity_env_cfg

BALL_LOST_DISTANCE = 1.0
EPISODE_LENGTH_S = 15.0
# Ball speed (m/s) above which moving the ball towards the goal earns nothing
# more. The velocity task's slow dribbling range topped out here too.
MAX_BALL_SPEED = 0.4
COMMAND_NAME = "command"
GOAL_NAME = "goal"


def dribble_mini_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = piplus_velocity_env_cfg(play=play)

  ##
  # Scene: add the ball. The robot plus ball touch more than the walking task's
  # contact budget, and dropped contacts look like sinking through the floor.
  ##
  cfg.scene.entities["ball"] = get_ball_cfg()
  cfg.sim.nconmax = 60

  ##
  # Command: a goal position for the ball, replacing the walking twist. One goal
  # per episode, 1-3 m away, within 45 degrees of where the robot faces.
  ##
  cfg.commands = {
    COMMAND_NAME: mdp.MultiCommandCfg(
      resampling_time_range=(0.0, 0.0),
      terms={
        GOAL_NAME: mdp.GoalPoseCommandCfg(
          entity_name="robot",
          debug_vis=True,
          resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
          radius_range=(1.0, 3.0),
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
  # twist's place in the observation.
  ##
  goal_params = {"command_name": COMMAND_NAME, "sub_command_name": GOAL_NAME}
  dribble_terms = {
    "command": ObservationTermCfg(func=mdp.goal_pos_b, params=goal_params),
    "ball_pos": ObservationTermCfg(func=mdp.ball_pos_b),
    "ball_vel": ObservationTermCfg(func=mdp.ball_vel_b),
  }
  cfg.observations["actor"].terms.update(dribble_terms)
  cfg.observations["critic"].terms.update(dribble_terms)

  ##
  # Events: robot faces +x, ball spawns 0.2-0.35 m ahead of it. No pushes.
  ##
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
  cfg.events.pop("push_robot", None)
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
  # Terminations: ball lost.
  ##
  cfg.terminations["ball_lost"] = TerminationTermCfg(
    func=mdp.ball_too_far,
    params={"max_distance": BALL_LOST_DISTANCE},
  )

  if not play:
    cfg.episode_length_s = EPISODE_LENGTH_S

  return cfg
