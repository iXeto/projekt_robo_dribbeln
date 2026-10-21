"""Mini dribble task: a smoke test, not a tuned policy.

Built on top of the PiPlus velocity task from mjlab_piplus, which already walks.
Added on top of it:

* a ball (RoboCup kid-size, from ``mjlab_piplus.objects``), placed just in front
  of the robot at every reset;
* the ball's position and velocity in the robot's yaw frame, for actor and
  critic (true state: no noise, no delay);
* two rewards: keep the ball in front of the feet, and move the ball at the
  commanded walking velocity;
* a termination when the ball is more than ``BALL_LOST_DISTANCE`` away.

The walking command is narrowed to slow, mostly-forward walking, and the robot
always spawns facing +x so that the ball's spawn point is in front of it.
"""

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
EPISODE_LENGTH_S = 10.0


def dribble_mini_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = piplus_velocity_env_cfg(play=play)

  ##
  # Scene: add the ball. The robot plus ball touch more than the walking task's
  # contact budget, and dropped contacts look like sinking through the floor.
  ##
  cfg.scene.entities["ball"] = get_ball_cfg()
  cfg.sim.nconmax = 60

  ##
  # Observations: ball state for actor and critic.
  ##
  ball_terms = {
    "ball_pos": ObservationTermCfg(func=mdp.ball_pos_b),
    "ball_vel": ObservationTermCfg(func=mdp.ball_vel_b),
  }
  cfg.observations["actor"].terms.update(ball_terms)
  cfg.observations["critic"].terms.update(ball_terms)

  ##
  # Command: slow, mostly forward. No standing, no in-place, no heading mode.
  ##
  twist = cfg.commands["twist"]
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.rel_walk_in_place_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.ranges.lin_vel_x = (0.15, 0.4)
  twist.ranges.lin_vel_y = (-0.05, 0.05)
  twist.ranges.ang_vel_z = (-0.3, 0.3)
  # The velocity curriculum would overwrite these ranges.
  cfg.curriculum = {}

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
  # Rewards: keep the walking terms, add the ball.
  ##
  cfg.rewards["ball_in_front"] = RewardTermCfg(
    func=mdp.ball_in_front,
    weight=1.5,
    params={"std": 0.3, "target_offset": 0.25},
  )
  cfg.rewards["ball_velocity"] = RewardTermCfg(
    func=mdp.ball_velocity_tracking,
    weight=1.5,
    params={"command_name": "twist", "std": 0.5},
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
