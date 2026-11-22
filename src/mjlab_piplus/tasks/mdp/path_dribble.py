"""Dribbeln entlang eines Pfads.

Der Ball soll einem vorgeplanten Pfad folgen. Die Pfade kommen aus derselben
A*-Bibliothek wie beim Path-Task (``path_following``): zufaellige Hindernisse,
und A* plant einen Weg mit Sicherheitsabstand darum herum. Die Hindernisse sind
virtuell -- sie werden nur angezeigt, die Physik kennt sie nicht.

Aufbau: ``PathDribbleCommand`` ist ein ``KickCommand`` (Ball platzieren,
Ballbeobachtung, Fussabstaende ... bleibt alles gleich). Neu ist nur, dass die
Dribbel-Richtung ``kick_dir_w`` nicht einmal gewuerfelt wird, sondern in jedem
Schritt aus dem Pfad kommt: vom Ball aus zu einem Punkt etwas weiter vorne auf
dem Pfad. Dadurch folgen alle bisherigen Dribbel-Belohnungen automatisch dem
Pfad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.utils.lab_api.math import wrap_to_pi

from mjlab_piplus.tasks.mdp.kick import KickCommand, KickCommandCfg
from mjlab_piplus.tasks.mdp.multi_command import resolve_command_term
from mjlab_piplus.tasks.mdp.path_following.path_command import (
  _add_box,
  load_or_generate_pathlib,
)
from mjlab_piplus.tasks.mdp.path_following.path_gen_astar import (
  OBSTACLE_KIND_CYLINDER,
  ObstacleWorldCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = [
  "PathDribbleCommand",
  "PathDribbleCommandCfg",
  "ball_on_path",
  "path_finished",
  "path_points",
]

_OBSTACLE_COLOR = (0.65, 0.35, 0.02, 0.5)


class PathDribbleCommand(KickCommand):
  cfg: PathDribbleCommandCfg

  def __init__(self, cfg: PathDribbleCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    # Die ganze Pfad-Bibliothek einmal laden (oder beim ersten Mal erzeugen und
    # auf die Festplatte cachen). Pfade beginnen alle im Ursprung.
    self.all_paths, _, self.all_obstacles = load_or_generate_pathlib(
      pathlib_size=cfg.pathlib_size,
      obstacle_world=cfg.obstacle_world,
      point_spacing=cfg.point_spacing,
      num_waypoints=cfg.num_waypoints,
      max_points=cfg.max_points,
      generation_seed=cfg.generation_seed,
      num_workers=cfg.num_workers,
      cache_dir=cfg.cache_dir,
      device=self.device,
    )
    # Pro Env ein Pfad (nur x, y) und seine Hindernisse.
    idx = torch.randint(0, cfg.pathlib_size, (self.num_envs,), device=self.device)
    self.paths = self.all_paths[idx, :, :2].clone()
    self.obstacles = self.all_obstacles[idx].clone()

    # Zustand pro Schritt.
    # path_index: welcher Pfadpunkt dem Ball am naechsten ist.
    self.path_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # ball_path_dist: wie weit der Ball vom Pfad weg ist (m).
    self.ball_path_dist = torch.zeros(self.num_envs, device=self.device)
    # path_done: Ball ist am Ende des Pfads angekommen.
    self.path_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # Die naechsten Pfadpunkte relativ zum Roboter (fuer die Beobachtung).
    self.path_points_b = torch.zeros(
      self.num_envs, 2 * cfg.num_path_observations, device=self.device
    )
    # Neue Pfade, die noch an den Ball verschoben werden muessen.
    self._path_pending = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _begin_ball_life(self, env_ids: torch.Tensor) -> None:
    """Neuer Ball = neuer Pfad. Verschoben wird er erst, wenn der Ball liegt."""
    super()._begin_ball_life(env_ids)
    if len(env_ids) == 0:
      return
    idx = torch.randint(0, self.cfg.pathlib_size, (len(env_ids),), device=self.device)
    self.paths[env_ids] = self.all_paths[idx, :, :2]
    self.obstacles[env_ids] = self.all_obstacles[idx]
    self.path_index[env_ids] = 0
    self.path_done[env_ids] = False
    self._path_pending[env_ids] = True

  def _apply_pending_placement(self) -> None:
    """Nach dem Platzieren des Balls den Pfad so verschieben, dass er am Ball beginnt."""
    placing = self._pending_place.clone()
    super()._apply_pending_placement()
    ids = (placing & self._path_pending).nonzero().flatten()
    if len(ids) == 0:
      return
    # _wrote_pos ist die Stelle, an die der Ball gerade gelegt wurde.
    offset = self._wrote_pos[ids].unsqueeze(1)
    self.paths[ids] += offset
    self.obstacles[ids, :, 1:3] += offset
    self._path_pending[ids] = False

  def _update_command(self) -> None:
    super()._update_command()  # Ballposition, Abstaende usw. wie beim Kick
    self._follow_path()

  def _follow_path(self) -> None:
    cfg = self.cfg
    n_points = self.paths.shape[1]
    rows = torch.arange(self.num_envs, device=self.device)

    # 1. Naechster Pfadpunkt zum Ball. Gesucht wird nur ein Stueck vor dem
    #    letzten, damit der Index nicht auf einen spaeteren Teil des Pfads
    #    springt, der zufaellig nah vorbeifuehrt.
    window = torch.arange(cfg.search_window, device=self.device)
    candidates = (self.path_index.unsqueeze(1) + window).clamp(max=n_points - 1)
    cand_xy = torch.gather(
      self.paths, 1, candidates.unsqueeze(-1).expand(-1, -1, 2)
    )  # (N, W, 2)
    dists = torch.norm(cand_xy - self.ball_pos_w.unsqueeze(1), dim=-1)  # (N, W)
    best = torch.argmin(dists, dim=1)
    self.path_index = candidates[rows, best]
    self.ball_path_dist = dists[rows, best]

    # 2. Ende erreicht? Der letzte Punkt des Pfads ist das Ziel.
    goal = self.paths[:, -1]
    self.path_done = torch.norm(goal - self.ball_pos_w, dim=1) < cfg.goal_radius

    # 3. Dribbel-Richtung = vom Ball zu einem Punkt etwas weiter vorne.
    ahead = (self.path_index + cfg.lookahead_points).clamp(max=n_points - 1)
    to_target = self.paths[rows, ahead] - self.ball_pos_w
    new_dir = torch.atan2(to_target[:, 1], to_target[:, 0])
    # Ganz am Ende ist der Zielpunkt fast erreicht: dann alte Richtung behalten.
    usable = torch.norm(to_target, dim=1) > 1e-3
    self.kick_dir_w = torch.where(usable, new_dir, self.kick_dir_w)
    # dir_err hat der KickCommand mit der alten Richtung berechnet: neu rechnen.
    self.dir_err = wrap_to_pi(self.kick_dir_w - self.robot.data.heading_w)

    # 4. Die naechsten Pfadpunkte relativ zum Roboter (x = vorne, y = links).
    offsets = (
      torch.arange(cfg.num_path_observations, device=self.device) + 1
    ) * cfg.observation_spacing
    obs_idx = (self.path_index.unsqueeze(1) + offsets).clamp(max=n_points - 1)
    pts = torch.gather(self.paths, 1, obs_idx.unsqueeze(-1).expand(-1, -1, 2))
    delta = pts - self.robot.data.root_link_pos_w[:, :2].unsqueeze(1)
    heading = self.robot.data.heading_w.unsqueeze(1)
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    local = torch.stack(
      [
        cos_h * delta[..., 0] + sin_h * delta[..., 1],
        -sin_h * delta[..., 0] + cos_h * delta[..., 1],
      ],
      dim=-1,
    )
    self.path_points_b = local.reshape(self.num_envs, -1)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    super()._debug_vis_impl(visualizer)  # oranger Richtungspfeil am Ball
    for batch in visualizer.get_env_indices(self.num_envs):
      # Pfad als kleine blaue Kugeln.
      xy = self.paths[batch, :: self.cfg.viz_path_stride].cpu().numpy()
      for x, y in xy:
        visualizer.add_sphere(
          np.array([x, y, 0.0]), radius=0.02, color=(0.0, 0.0, 1.0, 0.5)
        )
      # Hindernisse (nur angezeigt, nicht in der Physik).
      for kind, x, y, yaw, a, b, half_h in self.obstacles[batch].cpu().numpy():
        if kind < 0:
          continue
        if kind == OBSTACLE_KIND_CYLINDER:
          visualizer.add_cylinder(
            np.array([x, y, 0.0]),
            np.array([x, y, 2.0 * half_h]),
            radius=float(a),
            color=_OBSTACLE_COLOR,
          )
        else:
          _add_box(visualizer, x, y, yaw, a, b, 2.0 * half_h)


@dataclass(kw_only=True)
class PathDribbleCommandCfg(KickCommandCfg):
  # Pfad-Bibliothek (wie PathCommandCfg).
  pathlib_size: int = 2000
  """Wie viele Pfade einmal vorgeplant werden. 2000 reicht und ist schnell."""
  obstacle_world: ObstacleWorldCfg = field(default_factory=ObstacleWorldCfg)
  """Hindernisse und Planung, siehe path_gen_astar.py."""
  point_spacing: float = 0.01
  """Abstand zwischen zwei Pfadpunkten (m)."""
  num_waypoints: int = 4
  max_points: int = 1500
  """Punkte pro Pfad. 1500 x 0,01 m = 15 m maximale Pfadlaenge."""
  generation_seed: int = 0
  num_workers: int | None = None
  cache_dir: str | None = None

  # Pfad folgen.
  lookahead_points: int = 40
  """Wie weit vorne der Zielpunkt fuer die Richtung liegt (40 x 0,01 m = 0,4 m)."""
  search_window: int = 60
  """Wie viele Punkte vor dem letzten nach dem naechsten Punkt gesucht wird."""
  num_path_observations: int = 4
  """Wie viele kommende Pfadpunkte die Policy sieht."""
  observation_spacing: int = 25
  """Abstand der beobachteten Punkte (25 x 0,01 m = 0,25 m)."""
  goal_radius: float = 0.3
  """Ball naeher als das am Pfadende = geschafft."""
  viz_path_stride: int = 10
  """Nur jeden n-ten Pfadpunkt zeichnen (sonst wird der Viewer langsam)."""

  def build(self, env: ManagerBasedRlEnv) -> PathDribbleCommand:
    return PathDribbleCommand(self, env)


##
# Beobachtung, Belohnung, Abbruch.
##


def _path_term(
  env: ManagerBasedRlEnv, command_name: str, sub_command_name: str | None
) -> PathDribbleCommand:
  term, _ = resolve_command_term(env, command_name, sub_command_name)
  assert isinstance(term, PathDribbleCommand)
  return term


def path_points(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
) -> torch.Tensor:
  """Beobachtung: die naechsten Pfadpunkte relativ zum Roboter, (N, 2 * Anzahl)."""
  return _path_term(env, command_name, sub_command_name).path_points_b


def ball_on_path(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
  std: float = 0.3,
) -> torch.Tensor:
  """Belohnung in [0, 1]: 1, wenn der Ball genau auf dem Pfad liegt."""
  term = _path_term(env, command_name, sub_command_name)
  return torch.exp(-torch.square(term.ball_path_dist) / std**2)


def path_finished(
  env: ManagerBasedRlEnv,
  command_name: str,
  sub_command_name: str | None = None,
) -> torch.Tensor:
  """True, wenn der Ball am Ende des Pfads angekommen ist (Abbruch + Bonus)."""
  return _path_term(env, command_name, sub_command_name).path_done