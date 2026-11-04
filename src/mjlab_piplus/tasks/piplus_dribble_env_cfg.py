
import math

import mujoco
import torch
from mjlab.entity import EntityCfg
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
from mjlab_piplus.tasks.dribble_walk_action import WalkCommandActionCfg  # NEU

DELAY_MIN = 1
DELAY_MAX = 1
WALK_FREQUENCY = 1.75
BALL_RADIUS = 0.07   # an euren echten Ball anpassen (in Metern)
BALL_MASS = 0.1      # an euren echten Ball anpassen (in kg)

# NEU (5): Wo liegt das fertig trainierte Lauf-Gehirn? Pfad relativ zum Projektordner.
# Wenn das Lauf-Training fertig ist, hier den neuen Pfad eintragen.
WALK_POLICY_PATH = "logs/rsl_rl/piplus_velocity/2026-11-04_15-21-16/2026-11-04_15-21-16.onnx"

# NEU: NUR ZUM TESTEN! Mit --agent zero laeuft der Roboter dann mit 0.3 m/s geradeaus.
# Vor dem Dribbel-Training wieder auf (0.0, 0.0, 0.0) setzen!
TEST_COMMAND_OFFSET = (0.0, 0.0, 0.0)


##
# Ball
##


# NEU (1): Bauplan fuer den Ball.
# Leeres Modell -> Koerper "ball" -> frei beweglich (freejoint) -> Kugelform mit Gewicht und Reibung.
def get_ball_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint()  # Ball kann sich frei bewegen und rollen
  body.add_geom(
    name="ball_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[BALL_RADIUS, 0, 0],
    mass=BALL_MASS,
    rgba=[1.0, 0.5, 0.0, 1.0],      # orange
    condim=6,                        # noetig, damit Rollreibung wirkt
    friction=[0.8, 0.02, 0.01],      # sonst rollt der Ball ewig weiter
  )
  return spec


# NEU (1): Den Bauplan fuer mjlab verpacken + Startposition (30 cm vor dem Ursprung, auf dem Boden).
def get_ball_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_ball_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, BALL_RADIUS)),
  )


# NEU (3): Wird bei jedem Neustart einer Runde aufgerufen.
# Schaut, wo der Roboter steht und wohin er schaut, und legt den Ball 22-30 cm davor.
def reset_ball_in_front_of_robot(
  env,
  env_ids,
  distance_range: tuple[float, float] = (0.22, 0.30),
  lateral_range: tuple[float, float] = (-0.05, 0.05),
):
  """NEU: Legt den Ball beim Reset vor den Roboter, in seine Blickrichtung."""

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)

  robot = env.scene["robot"]
  ball = env.scene["ball"]
  n = len(env_ids)

  # Zufaellig: wie weit vor dem Roboter, wie weit seitlich versetzt
  dist = torch.empty(n, device=env.device).uniform_(*distance_range)
  lat = torch.empty(n, device=env.device).uniform_(*lateral_range)

  # Neue Roboterpose (aus reset_base) erst durchrechnen lassen, sonst lesen wir die alte
  env.sim.forward()

  # Wo steht der Roboter, wohin schaut er?
  robot_xy = robot.data.root_link_pos_w[env_ids, :2]
  heading = robot.data.heading_w[env_ids]
  cos_h, sin_h = torch.cos(heading), torch.sin(heading)

  # "vor dem Roboter" in Weltkoordinaten umrechnen
  x = robot_xy[:, 0] + cos_h * dist - sin_h * lat
  y = robot_xy[:, 1] + sin_h * dist + cos_h * lat
  z = torch.full_like(x, BALL_RADIUS)

  pos = torch.stack([x, y, z], dim=-1)
  quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(n, 1)
  vel = torch.zeros(n, 6, device=env.device)  # Ball liegt still

  ball.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=-1), env_ids)


# NEU (4): Hilfsfunktion. Rechnet aus, wo der Ball AUS SICHT DES ROBOTERS liegt.
# Ergebnis z. B. (0.25, 0.05) = "25 cm vor mir, 5 cm links". Brauchen wir spaeter auch fuer Belohnungen.
def ball_pos_in_robot_frame(env):
  """NEU: Wo liegt der Ball aus Sicht des Roboters? x = vorne, y = links (Meter)."""
  robot = env.scene["robot"]
  ball = env.scene["ball"]
  dx = ball.data.root_link_pos_w[:, 0] - robot.data.root_link_pos_w[:, 0]
  dy = ball.data.root_link_pos_w[:, 1] - robot.data.root_link_pos_w[:, 1]
  heading = robot.data.heading_w
  cos_h, sin_h = torch.cos(heading), torch.sin(heading)
  x = cos_h * dx + sin_h * dy
  y = -sin_h * dx + cos_h * dy
  return torch.stack([x, y], dim=-1)


# NEU (4): Prueft fuer jeden Roboter: Ist der Ball weiter als 50 cm weg? Dann True = Runde vorbei.
def ball_lost(env, max_distance: float = 0.5):
  """NEU: True, wenn der Ball weiter als max_distance vom Roboter weg ist."""
  return torch.norm(ball_pos_in_robot_frame(env), dim=-1) > max_distance


# NEU (8): Wie schnell rollt der Ball AUS SICHT DES ROBOTERS? (x = nach vorne, y = nach links, m/s)
# Gemessen relativ zum Roboter: Laeuft der Roboter genauso schnell wie der Ball, ist das 0.
def ball_vel_in_robot_frame(env):
  robot = env.scene["robot"]
  ball = env.scene["ball"]
  dvx = ball.data.root_link_lin_vel_w[:, 0] - robot.data.root_link_lin_vel_w[:, 0]
  dvy = ball.data.root_link_lin_vel_w[:, 1] - robot.data.root_link_lin_vel_w[:, 1]
  heading = robot.data.heading_w
  cos_h, sin_h = torch.cos(heading), torch.sin(heading)
  vx = cos_h * dvx + sin_h * dvy
  vy = -sin_h * dvx + cos_h * dvy
  return torch.stack([vx, vy], dim=-1)


##
# NEU (9): ZIEL. Wohin soll der Ball gedribbelt werden?
##


def _goal(env):
  """Zielpunkt (x, y) in der Welt, einer pro Roboter. Wird beim ersten Aufruf angelegt."""
  if not hasattr(env, "dribble_goal_w"):
    env.dribble_goal_w = torch.zeros(env.num_envs, 2, device=env.device)
  return env.dribble_goal_w


def reset_dribble_goal(
  env,
  env_ids,
  distance_range: tuple[float, float] = (2.0, 4.0),
  angle_range: tuple[float, float] = (-math.pi / 4, math.pi / 4),
):
  """NEU (9): Bei jedem Neustart ein neues Ziel: 2-4 m entfernt, grob vor dem Roboter (+-45 Grad).

  Muss NACH reset_base und reset_ball stehen (die Roboterpose ist dann schon aktuell).
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  goal = _goal(env)
  robot = env.scene["robot"]
  n = len(env_ids)
  dist = torch.empty(n, device=env.device).uniform_(*distance_range)
  angle = torch.empty(n, device=env.device).uniform_(*angle_range)
  heading = robot.data.heading_w[env_ids] + angle
  robot_xy = robot.data.root_link_pos_w[env_ids, :2]
  goal[env_ids, 0] = robot_xy[:, 0] + dist * torch.cos(heading)
  goal[env_ids, 1] = robot_xy[:, 1] + dist * torch.sin(heading)


def goal_pos_in_robot_frame(env):
  """NEU (9): Wo liegt das Ziel AUS SICHT DES ROBOTERS? (vorne / links, in Metern, max. 5 m)"""
  goal = _goal(env)
  robot = env.scene["robot"]
  dx = goal[:, 0] - robot.data.root_link_pos_w[:, 0]
  dy = goal[:, 1] - robot.data.root_link_pos_w[:, 1]
  heading = robot.data.heading_w
  cos_h, sin_h = torch.cos(heading), torch.sin(heading)
  x = cos_h * dx + sin_h * dy
  y = -sin_h * dx + cos_h * dy
  return torch.clamp(torch.stack([x, y], dim=-1), -5.0, 5.0)


def ball_goal_distance(env):
  """Abstand Ball <-> Ziel in Metern."""
  ball = env.scene["ball"]
  return torch.norm(_goal(env) - ball.data.root_link_pos_w[:, :2], dim=-1)


def goal_reached(env, radius: float = 0.3):
  """NEU (9): True, wenn der Ball naeher als 30 cm am Ziel ist = ERFOLG."""
  return ball_goal_distance(env) < radius


##
# NEU (10): BELOHNUNGEN fuers Dribbeln
##


def reward_ball_to_goal(env):
  """+ wenn der Ball Richtung Ziel rollt, - wenn er sich entfernt (m/s in Zielrichtung)."""
  ball = env.scene["ball"]
  to_goal = _goal(env) - ball.data.root_link_pos_w[:, :2]
  to_goal = to_goal / (torch.norm(to_goal, dim=-1, keepdim=True) + 1e-6)
  ball_vel = ball.data.root_link_lin_vel_w[:, :2]
  return torch.sum(ball_vel * to_goal, dim=-1)


def reward_ball_in_front(env, std: float = 0.15):
  """Nahe 1, wenn der Ball ca. 25 cm direkt vor dem Roboter liegt, sonst kleiner."""
  ball_b = ball_pos_in_robot_frame(env)
  target = torch.tensor([0.25, 0.0], device=env.device)
  dist_sq = torch.sum((ball_b - target) ** 2, dim=-1)
  return torch.exp(-dist_sq / std**2)


def reward_robot_faces_goal(env):
  """Nahe 1, wenn der Roboter zum Ziel schaut. Hilft, den Ball in die richtige Richtung zu schieben."""
  g = goal_pos_in_robot_frame(env)
  return g[:, 0] / (torch.norm(g, dim=-1) + 1e-6)


def reward_robot_to_goal(env):
  """NEU (12): + wenn der ROBOTER Richtung Ziel laeuft (m/s in Zielrichtung).
  Gibt dem Roboter von Anfang an einen Grund loszulaufen. Mit dem Ball vor den Fuessen
  schiebt er ihn dabei automatisch mit."""
  robot = env.scene["robot"]
  to_goal = _goal(env) - robot.data.root_link_pos_w[:, :2]
  to_goal = to_goal / (torch.norm(to_goal, dim=-1, keepdim=True) + 1e-6)
  robot_vel = robot.data.root_link_lin_vel_w[:, :2]
  return torch.sum(robot_vel * to_goal, dim=-1)


def ball_lost_flag(env):
  """True, wenn die Runde wegen Ballverlust endet."""
  return env.termination_manager.get_term("ball_lost")


def fell_over_flag(env):
  """True, wenn die Runde wegen Umfallen endet."""
  return env.termination_manager.get_term("fell_over")


def time_penalty(env):
  """NEU (11): Gibt jeden Schritt 1 zurueck. Mit negativem Gewicht = jeder Schritt kostet etwas.
  Wie im Paper: wer troedelt, verliert Punkte -> schnell zum Ziel lohnt sich."""
  return torch.ones(env.num_envs, device=env.device)


def failed(env):
  """True bei Fehlschlag: Ball verloren ODER umgefallen."""
  term = env.termination_manager
  return term.get_term("ball_lost") | term.get_term("fell_over")


def piplus_dribble_env_cfg(
  play: bool = False, static: bool = False
) -> ManagerBasedRlEnvCfg:
  foot_sites = ("l_foot", "r_foot")
  ##
  # Observations
  ##
  # NEU (8): Was das DRIBBEL-Gehirn sieht.
  # Vorher: Gelenkwinkel usw. (das braucht jetzt nur noch das Lauf-Gehirn, das holt sie sich selbst).
  # Jetzt:  vor allem der BALL.
  actor_terms = {
    # Wo liegt der Ball? (vorne / links, in Metern). Etwas Rauschen, weil die Kamera nicht perfekt ist.
    "ball_pos": ObservationTermCfg(
      func=ball_pos_in_robot_frame,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    # Wie schnell rollt der Ball weg bzw. auf mich zu?
    "ball_vel": ObservationTermCfg(
      func=ball_vel_in_robot_frame,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # Wie schnell drehe ich mich gerade? (Gyro)
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/gyro"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # Was habe ich zuletzt befohlen? (vorwaerts, seitwaerts, drehen)
    "actions": ObservationTermCfg(func=mdp.last_action),
    # NEU (9): Wo liegt das Ziel? (vorne / links, in Metern)
    "goal_pos": ObservationTermCfg(func=goal_pos_in_robot_frame),
  }
  # Der "Kritiker" hilft nur beim Training und darf mehr sehen (ohne Rauschen + eigene Geschwindigkeit).
  critic_terms = {
    "ball_pos": ObservationTermCfg(func=ball_pos_in_robot_frame),
    "ball_vel": ObservationTermCfg(func=ball_vel_in_robot_frame),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/gyro"},
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/local_linvel"},
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "goal_pos": ObservationTermCfg(func=goal_pos_in_robot_frame),
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

  # NEU (5): FAHRER statt direkter Gelenksteuerung.
  # Vorher: Policy gibt 12 Zahlen aus (eine pro Gelenk).
  # Jetzt:  Policy gibt 3 Zahlen aus (vorwaerts, seitwaerts, drehen).
  #         WalkCommandAction (dribble_walk_action.py) gibt sie ans Lauf-Gehirn weiter,
  #         das Lauf-Gehirn berechnet daraus die 12 Gelenkbefehle.
  actions: dict[str, ActionTermCfg] = {
    "walk_command": WalkCommandActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.1,
      use_default_offset=True,
      walk_policy_path=WALK_POLICY_PATH,
      command_offset=TEST_COMMAND_OFFSET,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    # NEU (6): Im Lauftraining wurde hier alle 3-8 Sekunden ein ZUFAELLIGER Befehl gewuerfelt.
    # Jetzt schreibt das Dribbel-Gehirn den Befehl hier rein. Darum alles Zufaellige aus:
    #   resampling 1e9  -> praktisch nie neu wuerfeln
    #   rel_... = 0.0   -> keine Sonderfaelle (stehen, nur geradeaus, Drehregler),
    #                      die dem Dribbel-Gehirn dazwischenfunken wuerden
    "twist": pi_mdp.UniformVelocityCommandWithStopCfg(
      entity_name="robot",
      resampling_time_range=(1e9, 1e9),
      debug_vis=True,
      heading_command=False,
      rel_standing_envs=0.0,
      rel_walk_in_place_envs=0.0,
      rel_heading_envs=0.0,
      rel_forward_envs=0.0,
      ranges=pi_mdp.UniformVelocityCommandWithStopCfg.Ranges(
        lin_vel_x=(-0.5, 0.8),
        lin_vel_y=(-0.4, 0.4),
        ang_vel_z=(-1.5, 1.5),
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
    # NEU (3): Ball beim Neustart vor den Roboter legen.
    # Muss NACH "reset_base" stehen: erst Roboter drehen, dann Ball davor legen.
    "reset_ball": EventTermCfg(
      func=reset_ball_in_front_of_robot,
      mode="reset",
      params={},
    ),
    # NEU (9): Neues Ziel bei jedem Neustart (muss NACH reset_ball stehen)
    "reset_goal": EventTermCfg(
      func=reset_dribble_goal,
      mode="reset",
      params={},
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
  # NEU (10): Belohnungen fuers DRIBBELN.
  # Die alten Lauf-Belohnungen (Gangbild, Fuesse, Gelenke ...) sind raus:
  # darum kuemmert sich das Lauf-Gehirn, das Dribbel-Gehirn kann daran nichts aendern.
  rewards = {
    # Hauptziel: Ball rollt Richtung Ziel
    # NEU (11): 2.0 -> 5.0, damit Schieben sich deutlich mehr lohnt als Stehen
    "ball_to_goal": RewardTermCfg(func=reward_ball_to_goal, weight=5.0),
    # Ball nah vor den Fuessen halten (nicht weit wegkicken)
    # NEU (11): 1.0 -> 0.2, sonst lohnt es sich, einfach neben dem Ball stehen zu bleiben
    "ball_in_front": RewardTermCfg(func=reward_ball_in_front, weight=0.2),
    # NEU (11): Jeder Schritt kostet etwas (Zeitdruck wie im Paper)
    "time_penalty": RewardTermCfg(func=time_penalty, weight=-0.5),
    # Zum Ziel schauen
    "faces_goal": RewardTermCfg(func=reward_robot_faces_goal, weight=0.3),
    # Grosser Bonus: Ball ist am Ziel angekommen
    "goal_reached": RewardTermCfg(func=goal_reached, weight=200.0),
    # NEU (12): Strafen getrennt.
    # Umfallen ist schlimm. Ball verlieren nur mittel, sonst traut sich der Roboter nicht an den Ball.
    "fell_over": RewardTermCfg(func=fell_over_flag, weight=-200.0),
    "ball_lost": RewardTermCfg(func=ball_lost_flag, weight=-30.0),
    # NEU (12): Roboter laeuft Richtung Ziel
    "robot_to_goal": RewardTermCfg(func=reward_robot_to_goal, weight=2.0),
    # Kleine Strafe fuer ruckartige Befehle (sonst zappelt der Roboter)
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
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
    # NEU (4): Ball zu weit weg -> Runde vorbei (Fehlschlag, wie im Paper)
    # NEU (12): 0.7 -> 1.0 m, der leichte Ball rollt beim Schieben schnell mal weiter weg
    "ball_lost": TerminationTermCfg(
      func=ball_lost,
      params={"max_distance": 1.0},
    ),
    # NEU (9): Ball am Ziel -> Runde vorbei (Erfolg)
    "goal_reached": TerminationTermCfg(func=goal_reached),
    # "root_height_below_minimum": TerminationTermCfg(
    #  func=mdp.root_height_below_minimum,
    #  params={"minimum_height": 0.25},
    # ),
  }

  ##
  # Curriculum
  ##
  # NEU (7): Kein Curriculum mehr. Im Lauftraining wurden damit die Zufallsbefehle langsam schneller.
  curriculum = {}

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
    entities={"robot": get_piplus_robot_cfg(), "ball": get_ball_cfg()},  # NEU (2): Ball in die Szene
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
      "reset_ball": events["reset_ball"],  # NEU (3): Ball-Reset auch im Play-Modus
      "reset_goal": events["reset_goal"],  # NEU (9): Ziel auch im Play-Modus
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