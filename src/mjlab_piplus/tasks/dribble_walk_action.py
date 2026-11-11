from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.tasks.velocity import mdp

from mjlab_piplus.tasks import mdp as pi_mdp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


##
# Lauf-Policy aus der .onnx-Datei in PyTorch nachbauen
##


# TEIL 1: Lauf-Gehirn oeffnen.
# Problem: Die .onnx-Datei kann nur 1 Roboter gleichzeitig rechnen ([1, 48]).
# Loesung: Wir lesen die gelernten Zahlen und Rechenschritte aus der Datei
#          und rechnen sie selbst mit PyTorch nach. Das geht fuer beliebig viele Roboter.
class OnnxAsTorch(torch.nn.Module):
  """Fuehrt ein einfaches ONNX-Netz (MLP + Normalisierung) direkt in PyTorch aus.

  Noetig, weil die exportierte .onnx-Datei nur 1 Roboter gleichzeitig kann ([1, 48]).
  So geht es fuer beliebig viele Roboter auf der GPU.
  """

  def __init__(self, path: str, device: str):
    super().__init__()
    import onnx
    from onnx import numpy_helper

    model = onnx.load(path)
    graph = model.graph

    self.consts: dict[str, torch.Tensor] = {}
    for init in graph.initializer:
      self.consts[init.name] = torch.tensor(
        numpy_helper.to_array(init), device=device, dtype=torch.float32
      )

    self.nodes = []
    for node in graph.node:
      attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
      if node.op_type == "Constant":
        self.consts[node.output[0]] = torch.tensor(
          numpy_helper.to_array(attrs["value"]), device=device, dtype=torch.float32
        )
        continue
      self.nodes.append((node.op_type, list(node.input), list(node.output), attrs))

    inputs = [i.name for i in graph.input if i.name not in self.consts]
    self.in_name = inputs[0]
    self.out_name = graph.output[0].name

    ops = sorted({op for op, *_ in self.nodes})
    print(f"[Dribble] Lauf-Policy geladen: {path}")
    print(f"[Dribble] ONNX-Bausteine: {ops}")

  @torch.no_grad()
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Jeder "op" ist ein Rechenschritt des Netzes (mal nehmen, addieren, Elu ...).
    # Wir fuehren sie der Reihe nach aus, genau wie in der .onnx-Datei.
    v = dict(self.consts)
    v[self.in_name] = x
    for op, ins, outs, a in self.nodes:
      i = [v[k] if k else None for k in ins]
      if op == "Gemm":
        A, B = i[0], i[1]
        if a.get("transA", 0):
          A = A.T
        if a.get("transB", 0):
          B = B.T
        y = a.get("alpha", 1.0) * (A @ B)
        if len(i) > 2 and i[2] is not None:
          y = y + a.get("beta", 1.0) * i[2]
      elif op == "MatMul":
        y = i[0] @ i[1]
      elif op == "Add":
        y = i[0] + i[1]
      elif op == "Sub":
        y = i[0] - i[1]
      elif op == "Mul":
        y = i[0] * i[1]
      elif op == "Div":
        y = i[0] / i[1]
      elif op == "Elu":
        y = F.elu(i[0], alpha=a.get("alpha", 1.0))
      elif op == "Relu":
        y = F.relu(i[0])
      elif op == "Tanh":
        y = torch.tanh(i[0])
      elif op == "Sigmoid":
        y = torch.sigmoid(i[0])
      elif op == "Identity":
        y = i[0]
      elif op == "Clip":
        lo = i[1] if len(i) > 1 and i[1] is not None else None
        hi = i[2] if len(i) > 2 and i[2] is not None else None
        y = torch.clamp(i[0], min=lo, max=hi)
      else:
        raise NotImplementedError(
          f"ONNX-Baustein '{op}' wird noch nicht unterstuetzt. Bitte an Claude schicken."
        )
      v[outs[0]] = y
    return v[self.out_name]


##
# Die Aktion
##


# TEIL 2: Der Fahrer.
# Baut auf JointPositionAction auf (mjlab), dadurch wissen wir schon,
# wie man die 12 Gelenke ansteuert. Wir aendern nur, WOHER die Gelenkbefehle kommen.
class WalkCommandAction(JointPositionAction):
  """Dribbel-Policy -> (vx, vy, wz) -> eingefrorene Lauf-Policy -> 12 Gelenke."""

  cfg: WalkCommandActionCfg

  # Einmal beim Start: alles vorbereiten.
  def __init__(self, cfg: WalkCommandActionCfg, env: ManagerBasedRlEnv):
    # Baut die normale Gelenksteuerung auf (12 Gelenke, scale 0.1, Standardpose).
    super().__init__(cfg=cfg, env=env)
    n = self.num_envs

    # Die Dribbel-Policy hat nur 3 Ausgaben statt 12.
    self._action_dim = 3
    self._raw_actions = torch.zeros(n, 3, device=self.device)

    self._cmd_scale = torch.tensor(cfg.command_scale, device=self.device)
    self._cmd_offset = torch.tensor(cfg.command_offset, device=self.device)
    self._cmd_low = torch.tensor(cfg.command_low, device=self.device)
    self._cmd_high = torch.tensor(cfg.command_high, device=self.device)

    # Eingefrorene Lauf-Policy.
    self.walk_policy = OnnxAsTorch(cfg.walk_policy_path, device=self.device)

    # Gedaechtnis fuer das Lauf-Gehirn:
    #   _walk_last_action: seine letzten 12 Gelenkbefehle (die will es als Eingabe sehen)
    #   _prev_sensor_obs:  Sensorwerte vom letzten Schritt (es ist an 1 Schritt Verzoegerung gewoehnt)
    #   _fresh:            True = Runde hat gerade neu angefangen, noch kein "letzter Schritt"
    self._walk_last_action = torch.zeros(n, 12, device=self.device)
    self._prev_sensor_obs = torch.zeros(n, 30, device=self.device)
    self._fresh = torch.ones(n, dtype=torch.bool, device=self.device)

    # Probelauf mit 2 Robotern: prueft, ob die Policy 48 rein / 12 raus kann.
    test_out = self.walk_policy(torch.zeros(2, 48, device=self.device))
    assert test_out.shape == (2, 12), f"Lauf-Policy gibt {test_out.shape} aus, erwartet (2, 12)"

  @property
  def action_dim(self) -> int:
    return 3

  # Was das Lauf-Gehirn "sieht".
  # Es wurde mit GENAU diesen 48 Zahlen in GENAU dieser Reihenfolge trainiert.
  # Bekommt es etwas anderes, macht es Unsinn. Darum bauen wir sie exakt nach.
  def _walk_obs(self) -> torch.Tensor:
    """Genau die 48 Beobachtungen aus actor_terms der Velocity-Task, gleiche Reihenfolge."""
    env = self._env
    sensor = torch.cat(
      [
        mdp.builtin_sensor(env, sensor_name="robot/gyro"),  # 3
        mdp.projected_gravity(env),  # 3
        mdp.joint_pos_rel(env, biased=True),  # 12
        mdp.joint_vel_rel(env),  # 12
      ],
      dim=-1,
    )
    # Beim Lauf-Training kamen diese Sensorwerte 1 Schritt verzoegert (DELAY = 1).
    delayed = torch.where(self._fresh[:, None], sensor, self._prev_sensor_obs)
    self._prev_sensor_obs = sensor.clone()
    self._fresh[:] = False

    return torch.cat(
      [
        delayed,  # 30
        self._walk_last_action,  # 12
        pi_mdp.phase(env, frequency=self.cfg.walk_frequency, command_name=self.cfg.command_name),  # 2
        mdp.generated_commands(env, command_name=self.cfg.command_name),  # 4
      ],
      dim=-1,
    )

  # In JEDEM Schritt: Befehl vom Dribbel-Gehirn -> Lauf-Gehirn -> Gelenke.
  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions

    # 1. Dribbel-Ausgabe -> Laufbefehl (auf den Bereich begrenzt, den die Lauf-Policy kennt)
    cmd = actions * self._cmd_scale + self._cmd_offset
    cmd = torch.max(torch.min(cmd, self._cmd_high), self._cmd_low)

    # 2. In den "twist"-Command schreiben, immer laufen (stand_flag = 0)
    twist = self._env.command_manager.get_term(self.cfg.command_name)
    twist.vel_command_b[:] = cmd
    twist.stand_flag[:] = 0.0

    # 3. Lauf-Policy fragen
    walk_actions = self.walk_policy(self._walk_obs())
    self._walk_last_action = walk_actions.clone()

    # 4. Wie JointPositionAction: Gelenkziel = Ausgabe * 0.1 + Standardpose
    self._processed_actions = walk_actions * self._scale + self._offset

  # Wenn eine Runde neu startet: Gedaechtnis loeschen,
  # sonst startet der Roboter mit Erinnerungen an die letzte Runde.
  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._walk_last_action[env_ids] = 0.0
    self._fresh[env_ids] = True


# TEIL 3: Einstellungen fuer den Fahrer (werden in piplus_dribble_env_cfg.py gesetzt).
@dataclass(kw_only=True)
class WalkCommandActionCfg(JointPositionActionCfg):
  walk_policy_path: str
  """Pfad zur .onnx-Datei der trainierten Lauf-Policy."""

  command_name: str = "twist"
  walk_frequency: float = 1.75

  command_scale: tuple[float, float, float] = (0.5, 0.3, 1.0)
  """Ausgabe der Dribbel-Policy * scale + offset = Laufbefehl (vx, vy, wz)."""
  command_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
  # Grenzen: Das Lauf-Gehirn hat nur diese Geschwindigkeiten geuebt.
  # Schnellere Befehle wuerden es ueberfordern (umfallen).
  command_low: tuple[float, float, float] = (-0.5, -0.4, -1.5)
  command_high: tuple[float, float, float] = (0.8, 0.4, 1.5)
  """Grenzen = Bereiche, in denen die Lauf-Policy trainiert wurde."""

  def build(self, env: ManagerBasedRlEnv) -> WalkCommandAction:
    return WalkCommandAction(self, env)