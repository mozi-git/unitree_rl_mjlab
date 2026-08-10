"""Unitree H2 INFIFORCE constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from src import SRC_PATH

##
# URDF and assets.
##

H2_INFIFORCE_URDF: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_h2_infiforce" / "urdf" / "H2.urdf"
)
assert H2_INFIFORCE_URDF.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, H2_INFIFORCE_URDF.parent / "meshes", meshdir)
  return assets


##
# Foot collision geometry.
##

# H2's URDF gives each foot a single collision *mesh*. Mesh-plane contact resolves to
# a handful of contact points that jump between mesh features as the foot rolls, so the
# normal force -- and with it the effective support polygon -- chatters. Every other
# humanoid in this zoo (G1, ELF3) instead lays a 7-capsule array across the sole, which
# gives smooth, distributed, well-conditioned contact. Do the same here: the collision
# mesh is renamed out of the `.*_collision` pattern so FULL_COLLISION's
# disable_other_geoms switches its contacts off, and these capsules take over.
#
# Layout measured from the foot mesh vertices in the *_ankle_pitch_link frame:
#   sole plane   z = -0.0543
#   heel..toe    x = [-0.0771, +0.1571]           (0.234 m)
#   inner..outer y = [-0.0010, +0.0762] (left)    (0.077 m, centered on +0.0376)
# Capsule centerlines are inset by the capsule radius so the swept surface stays inside
# that outline, and the heel/toe extents taper at the edges the way the sole does.

FOOT_CAPSULE_RADIUS = 0.01
_FOOT_SOLE_Z = -0.0543
_FOOT_CENTER_Y = 0.0376

# (y offset from the sole's lateral center, heel x, toe x) per capsule.
_FOOT_CAPSULES = (
  (-0.0286, -0.063, 0.141),
  (-0.0190, -0.065, 0.145),
  (-0.0095, -0.067, 0.147),
  (0.0000, -0.067, 0.147),
  (0.0095, -0.067, 0.147),
  (0.0190, -0.065, 0.145),
  (0.0286, -0.063, 0.141),
)


def _name_collision_geoms(spec: mujoco.MjSpec) -> None:
  for body in spec.bodies:
    collision_index = 1
    for geom in body.geoms:
      if geom.contype == 0 and geom.conaffinity == 0:
        continue
      if body.name in ("left_ankle_pitch_link", "right_ankle_pitch_link"):
        # Deliberately not a `*_collision` name, see _add_foot_capsules.
        side = "left" if body.name.startswith("left") else "right"
        geom.name = f"{side}_foot_hull{collision_index}"
      else:
        geom.name = f"{body.name}_{collision_index}_collision"
      collision_index += 1


def _add_foot_capsules(spec: mujoco.MjSpec) -> None:
  """Add the sole contact capsule array. Must run after _name_collision_geoms."""
  z = _FOOT_SOLE_Z + FOOT_CAPSULE_RADIUS
  for side, sign in (("left", 1.0), ("right", -1.0)):
    body = spec.body(f"{side}_ankle_pitch_link")
    for index, (y_offset, x_heel, x_toe) in enumerate(_FOOT_CAPSULES, start=1):
      y = sign * (_FOOT_CENTER_Y + y_offset)
      body.add_geom(
        name=f"{side}_foot{index}_collision",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=(FOOT_CAPSULE_RADIUS, 0.0, 0.0),
        fromto=(x_heel, y, z, x_toe, y, z),
        density=0.0,  # Link inertia comes from the URDF <inertial> tags.
      )


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(H2_INFIFORCE_URDF))
  spec.assets = get_assets(spec.meshdir)

  pelvis = spec.body("pelvis")
  if not pelvis.joints:
    pelvis.add_freejoint(name="floating_base_joint")

  spec.body("left_ankle_pitch_link").add_site(
    name="left_foot", pos=(0.05, 0.03, -0.03), size=(0.02,)
  )
  spec.body("right_ankle_pitch_link").add_site(
    name="right_foot", pos=(0.05, -0.03, -0.03), size=(0.02,)
  )
  # The IMU rides the pelvis, not the torso. H2.urdf carries mount frames for both
  # (`imu_in_torso_joint`, `imu_in_pelvis_joint`) and Unitree's own h2.xml wires its
  # sensors to a pelvis site, so pelvis is the one that matches the hardware. The
  # position below is the URDF's `imu_in_pelvis_joint` origin. It does not affect
  # `imu_ang_vel` -- angular velocity is common to every point of a rigid body -- but
  # `imu_lin_vel` reads v = v_body + omega x r, so the offset does matter there.
  spec.body("pelvis").add_site(
    name="imu_in_pelvis", pos=(-0.055, 0.0, -0.0589), size=(0.01,)
  )
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_in_pelvis",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_in_pelvis",
  )
  # Scoped to the torso subtree, not to pelvis, and the distinction inverts what the
  # `angular_momentum` reward pays for. `subtreeangmom` on pelvis -- the kinematic root --
  # is *whole-body* angular momentum about the whole-body CoM, and counter-rotating the
  # upper body against the swinging legs is precisely how you drive that toward zero. So
  # with a lower-body-only action space, where no arm swing is reachable, a penalty on the
  # pelvis reading was paying for the torso yaw oscillation it was meant to suppress. On
  # torso_link it measures only the 34.4 kg upper block's own angular momentum, which the
  # oscillation raises. Unitree's official h2.xml scopes it to torso_link for the same
  # reason.
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="torso_link",
  )
  _name_collision_geoms(spec)
  _add_foot_capsules(spec)

  return spec


##
# Actuator config.
##

ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

##
# Position-servo gains.
##

# The STIFFNESS_*/DAMPING_* rule above is rotor-referred: it places the *motor's* own
# natural frequency at NATURAL_FREQ and ignores the link inertia reflected at the joint.
# That is only self-consistent while the load is comparable to the armature. H2's leg
# links carry 100-200x their actuator's reflected inertia, so applied to the legs it
# produced a hip pitch closed loop at 0.70 Hz with zeta=0.14 -- a spring whose ring-down
# time, 1/(zeta*w_n) ~ 1.6 s, outlasts an entire gait cycle, leaving the policy no way
# to place a swing foot. (ELF3 escapes this only because its BXI85 armature happens to
# be 4.4x larger while its links are 2x lighter.)
#
# The gains below are instead solved from the joint-space inertia M_jj of the compiled
# model at KNEES_BENT_KEYFRAME, for an explicit load-referred bandwidth and damping:
#
#   kp = M_jj * (2*pi*f_n)**2        kd = 2 * zeta * sqrt(kp * M_jj)
#
#   joint             M_jj    f_n   zeta       kp      kd   tau_max   sat_err
#   hip_pitch       2.0917   1.90   0.50    298.1   24.97     360.0   69.2 deg
#   hip_roll        2.3777   1.80   0.50    304.1   26.89     360.0   67.8 deg
#   knee            0.4727   4.00   1.00    298.6   23.76     360.0   69.1 deg
#   waist_yaw       0.8314   1.90   0.50    118.5    9.93     120.0   58.0 deg
#   waist_pitch     2.6134   1.50   0.60    232.1   29.56     180.0   44.4 deg
#   waist_roll      3.2328   1.50   0.60    287.2   36.56     180.0   35.9 deg
#   shoulder_pitch  0.6135   2.60   0.70    163.7   14.03     130.0   45.5 deg
#   shoulder_roll   0.4198   2.60   0.70    112.0    9.60      60.0   30.7 deg
#   shoulder_yaw    0.0939   2.60   0.70     25.1    2.15      60.0  137.2 deg
#   elbow           0.1194   2.60   0.70     31.9    2.73      60.0  107.9 deg
#
# M_jj is read off the *articulated* model, so it includes the actuator armature on the
# diagonal -- that is what the closed loop actually sees, and what test_h2_constants
# recomputes. Refreshing it against the current H2.urdf left the leg and knee rows
# unchanged; the waist and arm rows moved by 1-9%. The arm rows use the mean of left and
# right values because one actuator config serves both sides: the URDF's inertials are
# not perfectly mirrored, and shoulder_yaw is the worst case at 7.7% inertia spread,
# which puts the realized f_n at 2.60 Hz +/- 1.8%. test_h2_constants allows 3% on the
# arms for that reason and stays at 2% everywhere else.
#
# `tau_max` is the URDF's declared effort limit for the joint, and `sat_err` the tracking
# error at which the servo hits it. Large transient errors saturating is physically
# correct and is not a reason to lower kp.
#
# CAUTION: `tau_max` used to come from the ACTUATOR_* block above rather than from the
# URDF, which put the legs at 88 / 139 Nm against the URDF's 360. That was not a case of
# hardware data disagreeing with a generic URDF field: the whole 5020 / 7520 / 4010 motor
# block is a verbatim copy of g1_constants.py, whose only provenance is a one-line
# "Motor specs (from Unitree)" comment -- i.e. those are *G1's* motors, and nothing
# establishes that H2 uses the same parts. G1 masses 33.3 kg against H2's 75.6 kg, so
# sharing a leg actuator between them is not plausible. The URDF's figures win by
# default, being at least H2's own file, but they have not been checked against a
# datasheet either and 360 Nm is high enough to be worth confirming.
#
# The armature values are still G1's, since the URDF declares none. That is a second
# order problem: armature is 0.5% of M_jj at the hip pitch and 5.3% at the knee, so even
# a 2x error there moves the solved gains by only a few percent. Real H2 motor specs
# would go in ROTOR_INERTIAS_* / GEARS_*, and the gain table would need re-solving.
#
# The arms are not in the velocity task's action space, so they are held by these servos
# alone -- at the old kp=14.25 H2's 12.7 kg of arms swung as 0.75 Hz / zeta=0.15 pendulums
# and pumped angular momentum into the torso that the policy could not cancel, hence they
# are retuned to ELF3's 2.6 Hz / zeta=0.7 as well.
#
# The remaining joints keep the rotor-referred values: their M_jj is small enough that
# those already land in a sane band (hip_yaw 2.70 Hz / 0.54, ankle_pitch 7.45 Hz / 1.49,
# ankle_roll 9.38 Hz / 1.88, wrists 4.99-7.57 Hz / 1.00-1.51, head 2.64-6.83 Hz /
# 0.53-1.37).
#
# tests/test_h2_constants.py recomputes M_jj from the compiled model and asserts the
# realized bandwidth and damping still match this table, so these numbers cannot drift
# out of sync with the geometry.

H2_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=298.1,
  damping=24.97,
  effort_limit=360.0,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
H2_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=360.0,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
H2_ACTUATOR_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=118.5,
  damping=9.93,
  effort_limit=120.0,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
H2_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=304.1,
  damping=26.89,
  effort_limit=360.0,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
H2_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=298.6,
  damping=23.76,
  effort_limit=360.0,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
H2_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint",),
  stiffness=163.7,
  damping=14.03,
  effort_limit=130.0,
  armature=ACTUATOR_5020.reflected_inertia,
)
H2_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_roll_joint",),
  stiffness=112.0,
  damping=9.60,
  effort_limit=60.0,
  armature=ACTUATOR_5020.reflected_inertia,
)
H2_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw_joint",),
  stiffness=25.1,
  damping=2.15,
  effort_limit=60.0,
  armature=ACTUATOR_5020.reflected_inertia,
)
H2_ACTUATOR_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_joint",),
  stiffness=31.9,
  damping=2.73,
  effort_limit=60.0,
  armature=ACTUATOR_5020.reflected_inertia,
)
H2_ACTUATOR_WRIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_roll_joint",),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=60.0,
  armature=ACTUATOR_5020.reflected_inertia,
)
# H2.urdf declares 10 Nm for the wrist pitch/yaw joints but 50 Nm for the head joints,
# so the single 4010 group these used to share cannot express both.
H2_ACTUATOR_WRIST_PITCH_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
  ),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=10.0,
  armature=ACTUATOR_4010.reflected_inertia,
)
H2_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "head_pitch_joint",
    "head_yaw_joint",
  ),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=50.0,
  armature=ACTUATOR_4010.reflected_inertia,
)
H2_ACTUATOR_WAIST_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint",),
  stiffness=232.1,
  damping=29.56,
  effort_limit=180.0,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
H2_ACTUATOR_WAIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_roll_joint",),
  stiffness=287.2,
  damping=36.56,
  effort_limit=180.0,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
H2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=66.88,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
H2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=STIFFNESS_4010 * 2,
  damping=DAMPING_4010 * 2,
  effort_limit=19.0,
  armature=ACTUATOR_4010.reflected_inertia * 2,
)

##
# Keyframe config.
##

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.05),
  joint_pos={
    ".*_hip_pitch_joint": -0.25,
    ".*_knee_joint": 0.55,
    ".*_ankle_pitch_joint": -0.3,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[0-9]+_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[0-9]+_collision$": 1},
  friction={r"^(left|right)_foot[0-9]+_collision$": (0.6,)},
)

##
# Final config.
##

H2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H2_ACTUATOR_HIP_PITCH,
    H2_ACTUATOR_HIP_ROLL,
    H2_ACTUATOR_HIP_YAW,
    H2_ACTUATOR_KNEE,
    H2_ACTUATOR_ANKLE_PITCH,
    H2_ACTUATOR_ANKLE_ROLL,
    H2_ACTUATOR_WAIST_YAW,
    H2_ACTUATOR_WAIST_PITCH,
    H2_ACTUATOR_WAIST_ROLL,
    H2_ACTUATOR_SHOULDER_PITCH,
    H2_ACTUATOR_SHOULDER_ROLL,
    H2_ACTUATOR_SHOULDER_YAW,
    H2_ACTUATOR_ELBOW,
    H2_ACTUATOR_WRIST_ROLL,
    H2_ACTUATOR_WRIST_PITCH_YAW,
    H2_ACTUATOR_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h2_infiforce_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=H2_ARTICULATION,
  )


# Action scale is the position-target offset range the policy can command around the
# default pose, so it has to be sized by the joint's range of motion over a gait -- NOT
# by the actuator's torque headroom. This used to be derived as
# `0.25 * effort_limit / stiffness`, which ties it to kp: with the retuned (up to 7x
# stiffer) gains above that rule would have collapsed the hip pitch range from 31 deg to
# 4.2 deg and put a walking gait kinematically out of reach. The values below are stated
# directly and preserve the ranges the policy already had.
H2_INFIFORCE_ACTION_SCALE: dict[str, float] = {
  # Lower body -- the velocity task's 15-joint action space.
  ".*_hip_pitch_joint": 0.55,
  ".*_hip_roll_joint": 0.35,
  ".*_hip_yaw_joint": 0.55,
  ".*_knee_joint": 0.35,
  ".*_ankle_pitch_joint": 0.59,
  ".*_ankle_roll_joint": 0.14,
  # The waist scales were carried over from the old `0.25 * effort_limit / stiffness`
  # rule and were never sized against a gait. At 0.55 rad the policy could twist the
  # torso 31 deg per step -- 2.6x the authority ELF3 has (waist_z 0.213) over a robot
  # whose above-waist block is 34.8 kg / 0.755 kg m^2 about the vertical, i.e. 46% of
  # H2's mass and 1.5x the yaw inertia of its own legs. With the arms outside the
  # action space, that block is the only sink the policy has for the yaw momentum the
  # swinging legs generate, so wide waist authority buys a visible torso oscillation
  # instead of a gait. Sized to ELF3's range, with a little headroom for H2's mass.
  "waist_yaw_joint": 0.25,
  "waist_pitch_joint": 0.25,
  "waist_roll_joint": 0.20,
  # Upper body -- unused by the 15-joint action space, present so that a full-body
  # action space still resolves a scale for every actuator.
  ".*_shoulder_pitch_joint": 0.35,
  ".*_shoulder_roll_joint": 0.35,
  ".*_shoulder_yaw_joint": 0.35,
  ".*_elbow_joint": 0.35,
  ".*_wrist_roll_joint": 0.35,
  ".*_wrist_pitch_joint": 0.20,
  ".*_wrist_yaw_joint": 0.20,
  "head_pitch_joint": 0.20,
  "head_yaw_joint": 0.20,
}
