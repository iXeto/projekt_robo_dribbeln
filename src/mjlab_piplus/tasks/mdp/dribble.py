from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import math
from mjlab.managers.reward_manager import RewardTermCfg

from mjlab_piplus.tasks.mdp.kick import KickCommand
from mjlab_piplus.tasks.mdp.multi_command import resolve_command_term

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

__all__ = ["ball_lost", "dribble_ball_proximity", "dribble_ball_velocity", "dribble_ball_still", "dribble_to_goal_pos", "dribble_to_goal_vel", "dribble_behind_ball", "dribble_ball_stop_at_goal", "dribble_ball_drift_penalty", "robot_facing_goal"]

def _kick(
  env: ManagerBasedRlEnv, command_name: str, sub_command_name: str | None
) -> KickCommand:
  term, _ = resolve_command_term(env, command_name, sub_command_name)
  assert isinstance(term, KickCommand)
  return term

def dribble_ball_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  std: float = 0.25,
) -> torch.Tensor:
  """``exp(-|v_ball - v_target|^2 / std^2)``: the ball rolls as commanded."""
  term = _kick(env, command_name, sub_command_name)
  ball_vel = term.ball.data.root_link_lin_vel_w[:, :2]
  err = torch.sum(torch.square(ball_vel - term.target_vel_w), dim=1)
  return torch.exp(-err / std**2)

def dribble_ball_proximity(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  std: float = 0.3,
) -> torch.Tensor:
  """``exp(-d^2 / std^2)``, d = ball to the nearest foot."""
  term = _kick(env, command_name, sub_command_name)
  nearest = term.foot_dist.min(dim=1).values
  return torch.exp(-torch.square(nearest) / std**2)

def ball_lost(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  max_dist: float = 1.2,
) -> torch.Tensor:
  """End the episode once the ball is more than ``max_dist`` from both feet."""
  term = _kick(env, command_name, sub_command_name)
  nearest = term.foot_dist.min(dim=1).values
  return (env.episode_length_buf > 5) & (nearest > max_dist)

def dribble_ball_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  vel_threshold: float = 0.1,
  std: float = 0.25,
) -> torch.Tensor:
  """``exp(-|v_ball - v_target|^2 / std^2)``: the ball rolls as commanded."""
  term = _kick(env, command_name, sub_command_name)
  speed = torch.norm(term.robot.data.root_link_lin_vel_w[:, :2], dim=1)
  return (speed < vel_threshold).float()

#Neuer
def dribble_to_goal_vel(
    env: ManagerBasedRlEnv,
    command_name: str,
    sub_command_name: str | None = None,
    max_speed: float = 3.0
  ) -> torch.Tensor:
    term = _kick(env, command_name, sub_command_name)

    ball_vel = term.ball.data.root_link_lin_vel_w[:, :2]
    ball_pos = term.ball.data.root_link_pos_w[:, :2]
    goals = dribble_to_goal_pos.goals
    if goals is None:
        return torch.zeros(env.num_envs, device=env.device)

    delta = goals - ball_pos
    dist = torch.norm(delta, dim=1)
    direction = delta / (dist.unsqueeze(1) + 1e-6)

    skalarprodukt =torch.sum(ball_vel * direction, dim=1)
    vel_reward = torch.clamp(skalarprodukt, min=0.0, max=max_speed) / max_speed

    return vel_reward
#Neuer
class dribble_to_goal_pos:
    goals: torch.Tensor | None = None
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.env = env
        dribble_to_goal_pos.goals = torch.zeros((env.num_envs, 2), device=env.device)
        
        self.reset(torch.arange(env.num_envs, device=env.device))

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        if len(env_ids) == 0:
            return
            
        robot_pos = self.env.scene["robot"].data.root_link_pos_w[env_ids, :2]
        
        distanz = torch.empty(len(env_ids), device=self.env.device).uniform_(3.0, 6.0)
        
        winkel = torch.empty(len(env_ids), device=self.env.device).uniform_(-math.pi, math.pi)
        
        offset_x = distanz * torch.cos(winkel)
        offset_y = distanz * torch.sin(winkel)
        
        dribble_to_goal_pos.goals[env_ids, 0] = robot_pos[:, 0] + offset_x
        dribble_to_goal_pos.goals[env_ids, 1] = robot_pos[:, 1] + offset_y

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        sub_command_name: str | None = None,
        std: float = 3.0,
    ) -> torch.Tensor:
        term = _kick(env, command_name, sub_command_name)
        
        ball_pos = term.ball.data.root_link_pos_w[:, :2]
        
        delta = dribble_to_goal_pos.goals - ball_pos
        dist = torch.norm(delta, dim=1)
        
        return torch.exp(-torch.square(dist) / std**2)

def dribble_behind_ball( # inspiration
    env: ManagerBasedRlEnv,
    command_name: str,
    sub_command_name: str | None = None,
) -> torch.Tensor:
    """Belohnt den Roboter dafür, dass er sich genau auf der Linie hinter dem Ball in Richtung Ziel aufstellt."""
    # 1. Daten über das Kick-Interface auslesen
    term = _kick(env, command_name, sub_command_name)

    # Positionen aus der Physik-Engine holen (nur X- und Y-Achse für 2D)
    robot_pos = term.robot.data.root_link_pos_w[:, :2]
    ball_pos = term.ball.data.root_link_pos_w[:, :2]
    
    # Notizbuch mit den Ziel-Koordinaten aus der Positions-Klasse auslesen
    goals = dribble_to_goal_pos.goals
    if goals is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 2. Vektoren berechnen
    # Vektor A: Richtungspfeil vom Roboter zum Ball ("Wohin muss der Roboter laufen?")
    robot_to_ball = ball_pos - robot_pos
    dist_robot_ball = torch.norm(robot_to_ball, dim=1)

    # Vektor B: Richtungspfeil vom Ball zum Tor ("Wohin soll der Ball rollen?")
    ball_to_goal = goals - ball_pos
    dist_ball_goal = torch.norm(ball_to_goal, dim=1)

    # 3. Vektoren auf die Länge 1.0 normieren (Einheitsvektoren)
    # .unsqueeze(1) bringt die 1D-Distanz (4096,) auf 2D (4096, 1), damit man 2D-Vektoren teilen kann.
    # + 1e-6 verhindert mathematische Abstürze (Division durch Null), falls Roboter oder Ball exakt am Ziel stehen.
    dir_robot_to_ball = robot_to_ball / (dist_robot_ball.unsqueeze(1) + 1e-6)
    dir_ball_to_goal = ball_to_goal / (dist_ball_goal.unsqueeze(1) + 1e-6)

    # 4. Skalarprodukt (Dot Product) zwischen beiden Richtungsvektoren berechnen
    # dim=1 zwingt PyTorch, das Skalarprodukt zeilenweise für jede der 4096 Welten zu berechnen.
    # Ergebnis ist der Cosinus des Winkels zwischen beiden Pfeilen:
    #   +1.0 = Roboter, Ball und Ziel liegen auf einer perfekten geraden Linie (Roboter steht RICHTIG dahinter).
    #    0.0 = Roboter steht seitlich zum Ball (90 Grad Winkel).
    #   -1.0 = Roboter steht zwischen Ball und Ziel (Roboter steht FALSCH und würde den Ball vom Ziel weg schieben).
    alignment = torch.sum(dir_robot_to_ball * dir_ball_to_goal, dim=1)

    # 5. Punktevergabe filtern
    # torch.clamp(..., min=0.0) schneidet alle negativen Werte ab. 
    # Steht der Roboter vor dem Ball, bekommt er 0.0 Punkte. 
    # Steht er dahinter, bekommt er je nach Präzision der Ausrichtung zwischen 0.0 und 1.0 Punkte.
    return torch.clamp(alignment, min=0.0)

def dribble_ball_stop_at_goal( #inspiration
    env: ManagerBasedRlEnv,
    command_name: str,
    sub_command_name: str | None = None,
    near_dist: float = 0.5,
    speed_std: float = 0.2,
) -> torch.Tensor:
    """Belohnt den Roboter dafür, dass der Ball in unmittelbarer Nähe des Ziels zur Ruhe kommt."""
    # 1. Daten über das Kick-Interface auslesen
    term = _kick(env, command_name, sub_command_name)

    # Position und Geschwindigkeitsvektor des Balls holen (nur X- und Y-Achse für 2D)
    ball_pos = term.ball.data.root_link_pos_w[:, :2]
    ball_vel = term.ball.data.root_link_lin_vel_w[:, :2]

    # Notizbuch mit den Ziel-Koordinaten aus der Positions-Klasse auslesen
    goals = dribble_to_goal_pos.goals
    if goals is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 2. Distanz vom Ball zum Ziel berechnen (Luftlinie in Metern)
    delta = goals - ball_pos
    dist = torch.norm(delta, dim=1)

    # 3. Absolute Ballgeschwindigkeit (Länge des Geschwindigkeitsvektors in m/s) berechnen
    ball_speed = torch.norm(ball_vel, dim=1)

    # 4. Ruhe-Reward via e-Funktion berechnen
    # Wenn der Ball stillsteht (ball_speed = 0.0), ergibt torch.exp(0.0) exakt 1.0 (Jackpot).
    # Je schneller der Ball rollt, desto steiler fällt der Wert in Richtung 0.0 ab.
    still_reward = torch.exp(-torch.square(ball_speed) / speed_std**2)

    # 5. Zielbereich-Filter (Maske):
    # (dist < near_dist) gibt True/False zurück.
    # .float() verwandelt das in 1.0 (innerhalb von 0.5m) oder 0.0 (außerhalb von 0.5m).
    is_near_goal = (dist < near_dist).float()

    # 6. Beide Signale multiplizieren:
    # Solange der Ball weit weg ist (0.0), ist das Ergebnis immer 0.0.
    # Erst wenn der Ball im 0.5m-Zielkreis liegt, greift der Ruhe-Reward.
    return is_near_goal * still_reward

def dribble_ball_drift_penalty( # inspiration
    env: ManagerBasedRlEnv,
    command_name: str,
    sub_command_name: str | None = None,
    max_dribble_dist: float = 0.8,
) -> torch.Tensor:
    """Bestraft den Roboter, wenn der Ball während des Dribbelns zu weit wegrollt."""
    term = _kick(env, command_name, sub_command_name)
    
    # Kürzeste Distanz zwischen den Füßen und dem Ball
    nearest_foot_dist = term.foot_dist.min(dim=1).values
    
    # Berechne die Überschreitung der gewünschten Dribbel-Distanz
    excess_dist = torch.clamp(nearest_foot_dist - max_dribble_dist, min=0.0)
    
    # Quadratische Strafe für zu weites Entfernen vom Ball
    return -torch.square(excess_dist)

def robot_facing_goal( # inspiration
    env: ManagerBasedRlEnv,
    command_name: str,
    sub_command_name: str | None = None,
) -> torch.Tensor:
    """Belohnt den Roboter dafür, dass sein Oberkörper in Richtung des Ziels blickt."""
    term = _kick(env, command_name, sub_command_name)

    robot_pos = term.robot.data.root_link_pos_w[:, :2]
    goals = dribble_to_goal_pos.goals
    if goals is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 1. Ausrichtung des Roboters aus der Rotationsmatrix holen (Vorwärts-Vektor X)
    # Forward-Vektor der Basis auf der XY-Ebene
    robot_forward = term.robot.data.root_link_state_w[:, 3:5] 

    # 2. Vektor vom Roboter zum Ziel
    robot_to_goal = goals - robot_pos
    dist = torch.norm(robot_to_goal, dim=1)
    dir_to_goal = robot_to_goal / (dist.unsqueeze(1) + 1e-6)

    # 3. Skalarprodukt: Blickt der Roboter zum Ziel?
    facing = torch.sum(robot_forward * dir_to_goal, dim=1)
    
    return torch.clamp(facing, min=0.0)