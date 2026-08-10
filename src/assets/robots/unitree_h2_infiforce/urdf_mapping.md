# `h2_constants.py` ↔ `H2.urdf` 对应关系

本文只回答一个问题：**`h2_constants.py` 里每个数值从哪来**——是 URDF 直接给的、是从 URDF 编译出的模型推导的、还是 URDF 里根本没有而由我们定的。

改 URDF 时对照本文最后一节「改了 URDF 之后要跟着动什么」。

设计动机和调参理由不在这里，见同目录任务配置下的 [`readme.md`](../../../tasks/velocity/config/h2_infiforce/readme.md) 与 [`adaptation.md`](../../../tasks/velocity/config/h2_infiforce/adaptation.md)。

---

## 0. 三类来源

| 类别 | 含义 | 改 URDF 后 |
| --- | --- | --- |
| **A｜URDF 直接给** | 数值原样抄自 URDF 标签 | 必须同步 |
| **B｜从 URDF 推导** | 由 URDF 编译出的模型算出来 | 必须重算 |
| **C｜URDF 没有** | URDF 不含此信息，我们自己定 | 通常不用动 |

一个关键事实决定了 C 类为什么这么多：

```
URDF 里 <transmission> 0 个，<gazebo> 0 个，<dynamics> 0 个
```

**`H2.urdf` 不包含任何驱动器信息**——没有 PD 增益、没有关节阻尼、没有摩擦、没有 armature。它只是运动学 + 惯量 + 视觉/碰撞网格 + 关节限位。所以整套伺服动力学都是我们加的。

---

## 1. 资产入口

| 常数 | 行 | 类别 | 对应 URDF |
| --- | --- | --- | --- |
| `H2_URDF` | 21 | A | 文件路径本身 |
| `get_assets()` | 27 | A | 解析 `<mesh filename="meshes/*.stl">`，把 `urdf/meshes/` 下的文件读进 `spec.assets` |

`get_spec()` 用 `mujoco.MjSpec.from_file()` 加载，URDF→MJCF 的转换由 MuJoCo 自带的 importer 完成，不是 mjlab 的逻辑。link/joint 层级、`<inertial>`、`<limit>`、mesh 引用全部原样进入 spec。

**URDF 引用 34 个 mesh**，`urdf/meshes/` 必须齐全，否则 `spec.compile()` 直接抛 `ValueError: Error opening file`。

---

## 2. `get_spec()` 的后处理（99–153 行）——**全部 C 类**

URDF 是固定基座的纯运动学描述，以下 5 项它都没有：

| 操作 | 行 | 说明 |
| --- | --- | --- |
| `pelvis.add_freejoint()` | 103–105 | URDF 无浮基关节。加上后整机可在世界系自由运动 |
| 足部 site `left_foot` / `right_foot` | 107–112 | `(0.05, ±0.03, -0.03)`，挂在 `*_ankle_pitch_link` 上。**手工标定值**，不是 URDF 里的点 |
| IMU site `imu_in_pelvis` | 118–120 | 位置 `(-0.055, 0, -0.0589)` — **这一个是 A 类**，取自 URDF 的 `imu_in_pelvis_joint` origin |
| 三个 sensor | 121–147 | `imu_ang_vel`(gyro) / `imu_lin_vel`(velocimeter) / `root_angmom`(subtreeangmom，作用体 `torso_link`) |
| 碰撞几何重命名 + 足底胶囊 | 68–96 | 见第 6 节 |

### 关于足部 site 的重要偏差

URDF 的足部碰撞体是一整块 mesh，没有定义「足底接触点」。我们的 site 放在 `z = -0.03`，而胶囊阵列的**足底面在 `z = -0.0543`**：

```
site 高出真实足底 0.0243 m
```

`foot_clearance` / `foot_swing_height` 两个奖励比较的是 **site 的世界系 z 原值**，不做足内偏移修正，所以任务配置里把 `target_height` 从共享默认的 0.1 改成了 **0.084** 来抵消这 24.3 mm。

> 对照：Unitree 官方 `h2.xml` 把 site 放在 `(0.04, ±0.037, -0.048)`，正好落在足底面上（胶囊中心线 −0.038、半径 0.01），所以官方的 `target_height=0.1` 就是字面意义的 0.1 m 离地。

### 关于 IMU

URDF 同时给了两个安装框架：

| URDF joint | parent | origin |
| --- | --- | --- |
| `imu_in_torso_joint` | `torso_link` | `(-0.06382, -0.06538, 0.29588)` |
| `imu_in_pelvis_joint` | `pelvis` | `(-0.055, 0, -0.0589)` |

我们用 **pelvis**，与官方 `h2.xml` 一致。位置不影响 `imu_ang_vel`（刚体上各点角速度相同），但影响 `imu_lin_vel`（读数是 `v + ω × r`）。

⚠️ **实机必须用 pelvis 那颗 IMU**，否则 sim2real 直接失效。

---

## 3. 执行器：URDF 给了什么

### A 类——力矩上限直接抄 URDF

`<limit effort="...">` 原样进入 `effort_limit`：

| 关节 | URDF effort | 代码位置 |
| --- | --- | --- |
| hip pitch / roll / yaw、knee | 360 | 282 / 303 / 289 / 310 |
| ankle pitch | 66.88 | 388 |
| ankle roll | 19 | 395 |
| waist yaw | 120 | 296 |
| waist roll / pitch | 180 | 381 / 374 |
| shoulder pitch | 130 | 317 |
| shoulder roll / yaw、elbow、wrist roll | 60 | 324 / 331 / 338 / 345 |
| wrist pitch / yaw | 10 | 354 |
| head pitch / yaw | 50 | 364 |

由 `tests/test_h2_constants.py::test_effort_limits_match_urdf` 逐条钉住。

> **腿部 360 Nm 存疑。** 它和第 4 节的电机型号推算对不上（7520-14 是 88 Nm、7520-22 是 139 Nm），只有两个踝关节在两个来源间一致。官方 `h2.xml` 也写 360，但两者可能同源于同一个通用 URDF 字段。建议对照数据手册确认。

### A 类——关节限位

`<limit lower/upper>` 由 MuJoCo importer 直接转成 `jnt_range`，代码里不重复声明。`H2_ARTICULATION.soft_joint_pos_limit_factor = 0.9`（C 类）把奖励用的软限位收到 90%。

### C 类——URDF 完全没有的部分

| 常数 | 行 | 来源 |
| --- | --- | --- |
| `ROTOR_INERTIAS_*` / `GEARS_*` | 159–180 | ⚠️ **从 `g1_constants.py` 原样拷贝**，见下 |
| `ARMATURE_*` | 161–182 | `reflected_inertia_from_two_stage_planetary()` 对上面两项的折算 |
| `ACTUATOR_*.velocity_limit` | 183–202 | 同上，来自 G1。**URDF 自己的 `<limit velocity>` 我们没用** |
| `NATURAL_FREQ` / `DAMPING_RATIO` | 204–205 | mjlab 通用规则的转子参照整定（10 Hz / ζ=2.0） |
| `STIFFNESS_*` / `DAMPING_*` | 207–215 | 由上面两项推出，仅供未重整定的关节使用 |

### ⚠️ 电机常数的实际出处

`5020` / `7520-14` / `7520-22` / `4010` 这四组型号连同它们的转子惯量、减速比、速度上限、力矩上限，是在 `387e8bb "Add h2 urdf"` 里随 H2 适配**从 `g1_constants.py` 逐字节拷贝**过来的。整条链上唯一的出处是 G1 那边的一行注释 `# Motor specs (from Unitree)`——**那是 G1 的电机参数，没有任何证据表明 H2 用同样的部件**。

而 G1 总质量 33.34 kg，H2 是 75.59 kg，**2.27 倍**。两台机器人共用同一套腿部执行器不合理。

由此：

- **力矩上限**：已改用 URDF 声明值（第 3 节）。原先的 88/139 Nm 是 G1 的腿部电机，对 H2 没有依据；URDF 的 360 Nm 虽然偏高，但至少是 H2 自己的文件声明的。仍建议对照 H2 数据手册确认。
- **`armature`**：URDF 不声明，所以只能沿用这套 G1 推算值。影响是**二阶的**——它在 `M_jj` 里的占比从 hip pitch 的 0.5% 到 knee 的 5.3%，即使差一倍，第 4 节解出的增益也只动几个百分点。
- **`velocity_limit`**：目前没有任何地方消费它，属于惰性数据。

拿到 H2 真实电机参数后，要改的是 `ROTOR_INERTIAS_*` / `GEARS_*`（→ `armature` → `M_jj` → 增益需重解）。

---

## 4. 增益推导链——**B 类**

除 hip yaw、两个踝、腕、头之外的关节，`kp` / `kd` 不来自电机，而是**从 URDF 的惯量解出来的**：

```
M_jj  ←  URDF <inertial> + 连杆几何，在 KNEES_BENT_KEYFRAME 位姿下编译后取关节空间惯量对角元
kp = M_jj * (2*pi*f_n)^2
kd = 2 * zeta * sqrt(kp * M_jj)
```

`f_n` / `zeta` 是我们指定的设计目标（C 类），`M_jj` 是 URDF 决定的（B 类）。

**`M_jj` 必须从「已配好执行器的」模型上读**，因为 `mj_fullM` 的对角线含 `armature`，而那正是闭环实际看到的量：

```python
m = Entity(get_h2_robot_cfg()).spec.compile()   # 对
m = get_spec().compile()                        # 错，缺 armature
```

用错会让手臂 `kp` 偏低约 4%，测试会直接抓出来（曾经就抓到过）。

当前值见 `h2_constants.py` 235–245 行的表格。左右臂共用一套 actuator 配置，而 URDF 的左右 `<inertial>` 并非精确镜像，所以臂部用左右 `M_jj` 的均值；shoulder yaw 最差，惯量差 7.7%，实现的 `f_n` 落在 2.60 Hz ±1.8%——测试对臂部放宽到 3%，其余保持 2%。

---

## 5. `H2_ACTION_SCALE`（475 行）——**C 类**

历史上是 `0.25 * effort_limit / stiffness`（上游至今仍这么算），现在**直接写死**。理由：它是策略能在默认位姿附近下发的位置目标偏移范围，应当由该关节在一个步态周期内需要的运动幅度决定，**而不是由力矩裕度决定**。沿用旧规则的话，配上第 4 节重整定后的高 `kp`，hip pitch 的可用范围会从 31° 塌到 4.2°，步态在运动学上就不可达了。

所以这一项**和 URDF 无关**，改 URDF 不需要动它。

---

## 6. 碰撞几何

| | 类别 | 说明 |
| --- | --- | --- |
| `_name_collision_geoms()` (68) | A→C | URDF 的碰撞 geom 是匿名的，这里给稳定名字。足部的两块 mesh 被改名为 `*_foot_hull{n}`，**故意落在 `.*_collision` 之外** |
| `_FOOT_CAPSULES` (57) | C | 7 根胶囊的布局，从足部 mesh 顶点手工量出来的 |
| `FOOT_CAPSULE_RADIUS`=0.01、`_FOOT_SOLE_Z`=−0.0543、`_FOOT_CENTER_Y`=0.0376 | C | 同上 |
| `FULL_COLLISION` (425) | C | `condim` / `priority` / `friction` |

URDF 的足部碰撞体是**单块 mesh**。mesh-plane 接触只解算出少数几个点，且随脚掌滚动在 mesh 特征间跳变，法向力和有效支撑多边形都会抖。所以改名把它排除掉，换成 7 根胶囊的阵列（与 G1 / ELF3 一致）。胶囊 `density=0`，足部惯量仍来自 URDF 的 `<inertial>`。

> 官方 `h2.xml` 里 `foot1..7_collision` 是**原生就有的**，说明 Unitree 本来就是这么设计的，我们等于从 URDF 侧重建了一遍。两者的足底面高度不同：官方 −0.038（中心线）、我们 −0.0543。

---

## 7. `KNEES_BENT_KEYFRAME`（407 行）——**C 类**

URDF 不含初始位姿。`pos=(0, 0, 1.05)` 与各关节角是我们定的，`z` 需与实际站立高度自洽（见 `adaptation.md` 第 4 节的实测校验）。

> 官方 `h2.xml` 的 pelvis 起始高度是 1.03，且 shoulder pitch 0.35 / elbow 0.87 / shoulder roll ±0.18，与我们的 0.2 / 0.6 / ±0.2 不同。

---

## 8. 测试覆盖

`tests/test_h2_constants.py` 把上述对应关系钉死，共 32 项：

| 测试 | 钉住的关系 |
| --- | --- |
| `test_gains_are_load_referred` | 第 4 节：从编译模型重算 `M_jj`，断言实现的 `f_n` / `zeta` 仍等于文档表 |
| `test_effort_limits_match_urdf` | 第 3 节：`effort_limit` 必须等于 URDF 声明值 |
| `test_imu_is_mounted_on_the_pelvis` | 第 2 节：两个 IMU sensor 必须挂在 pelvis 的 site 上 |
| `test_foot_collision_geoms` | 第 6 节：14 根胶囊、`condim=3`、`priority=1`、`friction=0.6` |
| `test_foot_hull_mesh_does_not_collide` | 第 6 节：原 mesh 的接触必须已关闭 |
| `test_foot_capsules_add_no_mass` | 第 6 节：胶囊 `density=0`，足部惯量仍来自 URDF `<inertial>` |
| `test_gravity_hold_within_effort_limits` | 保持标称位姿所需力矩不超限 |
| `test_action_scale_covers_every_actuator` | `H2_ACTION_SCALE` 覆盖到每一个执行器，无遗漏 |
| `test_h2_entity_creation` | 整份配置能构造出 Entity |

**改完 URDF 先跑这个测试**，它会告诉你哪些推导值飘了。

---

## 9. 改了 URDF 之后要跟着动什么

按必须程度排序：

1. **mesh 齐全性** — 新 URDF 引用的 `.stl` 必须都在 `urdf/meshes/`，否则编译直接失败。
2. **跑 `tests/test_h2_constants.py`** — 增益和力矩上限的漂移会在这里暴露。
3. **`<limit effort>` 变了** → 同步第 3 节表格里的 `effort_limit`，并更新测试的 `URDF_EFFORT_LIMITS`。
4. **`<inertial>` 变了** → `M_jj` 变 → 按第 4 节公式重解 `kp`/`kd`，更新 `h2_constants.py` 235–245 行的表格和 `readme.md` 的增益表。注意用 `Entity(...)` 编译而不是 `get_spec()`。
5. **足部几何变了** → 重新量 `_FOOT_CAPSULES` / `_FOOT_SOLE_Z`，并重新标定 site 相对足底的高度差，同步任务配置里的 `target_height`。
6. **关节数量或名称变了** → 动作空间、观测布局、部署接口全部受影响。注意 fixed 关节挂的连杆会被 MuJoCo 合并进父 body，**不产生新自由度**，这类新增（如 `left_hand_link`、`camera`）只改变质量分布，不改变接口。
7. **IMU 安装位置变了** → 同步第 2 节的 site 位置。

不需要动的：`H2_ACTION_SCALE`（第 5 节）、`NATURAL_FREQ`/`DAMPING_RATIO`、电机型号相关的 `ROTOR_INERTIAS_*` / `GEARS_*` / `velocity_limit`。
