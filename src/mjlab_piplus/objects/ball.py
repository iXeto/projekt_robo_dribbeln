"""A free-rolling soccer ball, as a scene entity.

Parameters follow the RoboCup kid-size field model in bit-bots/bitbots_main
(``bitbots_mujoco_sim/xml/kid_field.xml``): a 7 cm sphere at 205 g with a full
friction cone and a stiff contact. The texture there is a cube map loaded from
PNGs; we skip it and use a flat colour, since nothing in training reads pixels.
"""

import mujoco
from mjlab.entity import EntityCfg

# Kid-size ball: 14 cm across, 205 g.
BALL_RADIUS = 0.07
BALL_MASS = 0.205

# (sliding, torsional, rolling). The small rolling term is what stops the ball
# from coasting forever once it is moving.
BALL_FRICTION = (0.4, 0.01, 0.01)

# Negative solref is a direct (stiffness, damping) pair rather than
# (timeconst, dampratio): a hard, lively contact so the ball actually bounces
# off a foot instead of sinking into it.
BALL_SOLREF = (-5000.0, -20.0)


def get_ball_spec(
  radius: float = BALL_RADIUS,
  mass: float = BALL_MASS,
  rgba: tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint(name="ball_root")
  body.add_site(name="ball_center") # NEU -> Ball mitte als messpunkt für den Pfad
  body.add_geom(
    name="ball_collision",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius, 0.0, 0.0),
    mass=mass,
    friction=BALL_FRICTION,
    # condim=6 gives torsional and rolling friction on top of sliding, which a
    # sphere needs to roll and spin down rather than slide frictionlessly.
    condim=6,
    # priority=1 makes this geom's friction/solref win the contact-parameter
    # mix against the floor and the robot, instead of being averaged with them.
    priority=1,
    solref=BALL_SOLREF,
    rgba=rgba,
  )
  return spec


def get_ball_cfg(
  radius: float = BALL_RADIUS,
  mass: float = BALL_MASS,
) -> EntityCfg:
  """Scene entity for the ball, resting on the ground at the world origin.

  It is a floating-base, non-articulated entity: one body, one freejoint, one
  geom. Place it per episode with a ``reset_root_state_uniform`` event.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, radius),  # Resting on the plane, not dropped onto it.
      # The freejoint is the only joint, so there is nothing for the joint
      # dicts to match; leaving them at the defaults keeps the entity builder
      # from looking for a keyframe that this spec does not have.
      joint_pos={},
      joint_vel={},
    ),
    spec_fn=lambda: get_ball_spec(radius=radius, mass=mass),
  )
