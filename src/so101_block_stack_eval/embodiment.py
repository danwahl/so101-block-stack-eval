"""The block-stack cell as an Inspect Robots embodiment, commanded by where the grip goes.

An agent names a point in space, the tool's pitch and the jaw's yaw, and how far open the
jaws are; a closed-form IK turns that into joint targets. The commanded point is the grip
point: where a block's centre sits once the jaws close on it, so a grasp is a move to the
block followed by a close, at any pitch.

The arm has five joints, so those five numbers fix it. Pitch is the IK's tilt, 0 level and
pi/2 straight down; it tips the tool in the arm's own vertical plane, never sideways
across it.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
from inspect_robots.embodiment import (
    PRIVILEGED_SUCCESS,
    RESETTABLE,
    EmbodimentBase,
    EmbodimentInfo,
)
from inspect_robots.scene import Scene
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)
from inspect_robots.types import Action, Observation, StepResult
from inspect_robots_agent import CustomTool, ToolError

from . import env  # noqa: F401  (registers the env)
from .kinematics import BASE_P, PAN_ORIGIN, ROLL_ZERO, ik, tool_axes
from .robot import BASE_POSE
from .spec import (
    ARM_JOINTS,
    BLOCK_NAMES,
    CAMERAS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    IMAGE_SIZE,
    LIMITS,
)

ENV_ID = "SO101BlockStack-v1"
LABELS = ("x", "y", "z", "yaw", "pitch", "gripper")
CONTROL_HZ = 10.0

# The box the agent can name: the arm's whole reach above the table, every yaw, and pitch
# from pointing up to pointing back at the base. Most of it is out of reach; `check_motion`
# says where.
LOW = np.array([-0.10, -0.45, 0.0, -math.pi, -math.pi / 2, 0.0])
HIGH = np.array([0.50, 0.45, 0.45, math.pi, math.pi, 1.0])
# A step's worth of motion at 10 Hz: 5 cm/s, half a radian a second, and a second from
# open to shut.
MAX_STEP = (0.005, 0.005, 0.005, 0.05, 0.05, 0.1)

# The most any joint may move in one step, as on the hardware.
MAX_JOINT_STEP = math.radians(10.0)

# The grip point from the TCP: just below it, so the jaw tips clear the table, and along the
# jaw axis, seating the block against the fixed jaw.
GRASP_DZ = -0.003
GRASP_LATERAL = 0.020


_BASE = np.asarray(BASE_POSE.p, np.float64)
_LO, _HI = (torch.as_tensor(LIMITS[:5, i], dtype=torch.float64) for i in (0, 1))

# The top camera, in the base frame: an ideal pinhole at TOP_CENTRE, tipped TOP_TILT_DEG
# from straight down toward +x, its image right along -y, as the sim builds it.
TOP_F = 546.0
TOP_CENTRE = (0.0335, 0.116, 0.5503)
TOP_TILT_DEG = 25.0
_TILT = math.radians(TOP_TILT_DEG)
TOP_K = np.array([[TOP_F, 0.0, 240.0], [0.0, TOP_F, 240.0], [0.0, 0.0, 1.0]])
# Columns: the camera's x (image right), y (image down) and z (view) axes, then its centre.
TOP_CAM_TO_BASE = np.array(
    [
        [0.0, -math.cos(_TILT), math.sin(_TILT), TOP_CENTRE[0]],
        [-1.0, 0.0, 0.0, TOP_CENTRE[1]],
        [0.0, -math.sin(_TILT), -math.cos(_TILT), TOP_CENTRE[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


DOCS = f"""\
- Frame: the arm's base, on the table top. Metres. +x points away from the base across \
the table, +y to the arm's left, +z up.
- x, y, z place the grip point: where a block's centre sits once the jaws close on it.
- pitch is the tool's angle below level: 0 points it level, the way the arm reaches, \
and pi/2 straight down; past pi/2 it tips back toward the base, and below 0 up. The arm \
has five joints, so the tool cannot tip sideways.
- yaw turns the wrist. Pointing straight down, it is the heading of the axis the jaws \
close along, from +x toward +y.
- gripper is 1 fully open (jaws 57 mm apart) and 0 shut (22 mm).
- A move is refused, with the point where it fails, if it leaves the arm's reach or \
would turn a joint more than {math.degrees(MAX_JOINT_STEP):.0f} degrees in one step.
- Blocks are 30 mm cubes.
- Cameras: both images are 480 x 480. "wrist" rides on the gripper, looking along the \
jaws. "top" is fixed above the base, looking down at the table: the top of its image is \
far from the base and its right is -y.
"""


def solve(targets):
    """Joint targets for ``(n, 6)`` actions, and which of them the arm can hold.

    The grip point sits off the TCP along the jaw and the tool, whose directions depend on
    the pan the IK will pick for that TCP; a second pass, knowing it, lands within a few
    microns, the offset being a tenth of the reach.
    """
    targets = np.array(targets, np.float64)
    grip = torch.as_tensor(targets[:, :3] + _BASE)
    yaw = torch.as_tensor(targets[:, 3])
    tilt = torch.as_tensor(targets[:, 4])
    tcp = grip
    for _ in range(3):
        reach = tcp - BASE_P - PAN_ORIGIN
        jaw, _, tool = tool_axes(torch.atan2(reach[:, 1], reach[:, 0]), yaw, tilt)
        tcp = grip + GRASP_LATERAL * jaw - GRASP_DZ * tool
    q, ok = ik(tcp, yaw, tilt)
    ok = ok & (q >= _LO).all(-1) & (q <= _HI).all(-1)
    width = GRIPPER_CLOSED + targets[:, 5:] * (GRIPPER_OPEN - GRIPPER_CLOSED)
    return np.concatenate([q.numpy(), width], -1), ok.numpy()


def read_pose(tcp, quat, qpos):
    """The commanded quantities back from the arm: grip point, yaw, pitch, opening."""
    w, x, y, z = quat
    jaw = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    tool = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    grip = tcp - GRASP_LATERAL * jaw + GRASP_DZ * tool - _BASE
    pan, lift, elbow, flex, roll = qpos[:5]
    yaw = (roll - pan - ROLL_ZERO + math.pi) % (2 * math.pi) - math.pi
    pitch = lift + elbow + flex
    opening = (qpos[5] - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED)
    return np.array([*grip, yaw, pitch, opening])


def check_motion(waypoints, start=None):
    """Why a move cannot be made, or None.

    A waypoint must solve inside the joint limits, and no joint may move more than
    ``MAX_JOINT_STEP`` from the one before, ``start`` being the joints the arm is at, from
    which the physical arm's step cap also measures. Close to the pan axis a small move of
    the grip can need a large turn of the base, which that cap would clip mid-motion. An
    unreachable target is reported over any earlier failure on the way.
    """
    q, ok = solve(waypoints)
    if start is not None:
        q = np.concatenate([start[None], q])
    moves = np.abs(np.diff(q[:, :5], axis=0)) if len(q) > 1 else np.zeros((0, 5))
    moves = np.concatenate([np.zeros((len(ok) - len(moves), 5)), moves])
    jump = moves.max(-1)
    bad = ~ok | (jump > MAX_JOINT_STEP)
    if not bad.any():
        return None
    first = int(np.argmax(bad)) if ok[-1] else len(ok) - 1
    x, y, z = waypoints[first, :3]
    where = "the target" if first == len(ok) - 1 else f"step {first + 1} of {len(ok)}"
    if ok[first]:
        joint = ARM_JOINTS[int(np.argmax(moves[first]))]
        return (
            f"refused at {where} (x={x:.3f}, y={y:.3f}, z={z:.3f}): {joint} would move "
            f"{math.degrees(jump[first]):.0f} degrees in one step "
            f"(limit {math.degrees(MAX_JOINT_STEP):.0f})."
        )
    return (
        f"out of reach at {where} (x={x:.3f}, y={y:.3f}, z={z:.3f}, "
        f"yaw={waypoints[first, 3]:.2f}, pitch={waypoints[first, 4]:.2f})."
    )


def locate_points(arguments, observation):
    """Where top-camera pixels fall on a horizontal plane, in the base frame.

    Each pixel's ray, from the camera's centre through the pixel, meets the plane at the
    given height.
    """
    points, height = arguments.get("points"), arguments.get("height")
    if not isinstance(height, (int, float)) or not 0.0 <= height < TOP_CENTRE[2]:
        raise ToolError(f"height must be a number from 0 to {TOP_CENTRE[2]:.2f} m")
    if (
        not isinstance(points, list)
        or not points
        or not all(isinstance(p, list) and len(p) == 2 for p in points)
    ):
        raise ToolError("points must be a non-empty list of [u, v] pixel pairs")
    lines = []
    for u, v in points:
        if not all(isinstance(c, (int, float)) and 0 <= c <= IMAGE_SIZE for c in (u, v)):
            raise ToolError(f"pixel [{u}, {v}] is outside the {IMAGE_SIZE} x {IMAGE_SIZE} image")
        ray = TOP_CAM_TO_BASE[:3, :3] @ np.linalg.solve(TOP_K, [u, v, 1.0])
        x, y, z = np.asarray(TOP_CENTRE) + (height - TOP_CENTRE[2]) / ray[2] * ray
        lines.append(f"[{u}, {v}] -> x {x:.3f}, y {y:.3f}, z {z:.3f}")
    return "\n".join(lines)


LOCATE_POINTS = CustomTool(
    name="locate_points",
    description=(
        "Base-frame positions, in metres, of pixels in the top camera's image, where each "
        "pixel's line of sight meets a horizontal plane at the given height."
    ),
    parameters={
        "type": "object",
        "properties": {
            "points": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
                "description": (
                    f"Pixels as [u, v]: u to the right and v down from the top-left corner "
                    f"of the {IMAGE_SIZE} x {IMAGE_SIZE} image."
                ),
            },
            "height": {
                "type": "number",
                "description": "Height of the plane in metres; 0 is the table.",
            },
        },
        "required": ["points", "height"],
    },
    run=locate_points,
)


class SimEmbodiment(EmbodimentBase):
    """One ``SO101BlockStack-v1`` env, reset per scene to that scene's layout."""

    def __init__(self):
        # One env steps faster on the CPU, where a joint write also lands without a flush.
        self._env = gym.make(ENV_ID, num_envs=1, obs_mode="rgb", sim_backend="physx_cpu").unwrapped
        self._q = None
        self._joints = None
        self._instruction = None
        state = StateSpec(
            fields=(
                StateField("eef_state", (6,), "m,m,m,rad,rad,open"),
                StateField("joint_pos", (6,), "rad"),
            )
        )
        self.info = EmbodimentInfo(
            name="so101_sim",
            action_space=Box(
                (6,),
                LOW,
                HIGH,
                ActionSemantics(
                    control_mode="eef_abs_pose",
                    rotation_repr="none",
                    gripper="continuous",
                    frame="base",
                    dim_labels=LABELS,
                    max_step=MAX_STEP,
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(CameraSpec(name, IMAGE_SIZE, IMAGE_SIZE) for name in CAMERAS),
                state=state,
            ),
            control_hz=CONTROL_HZ,
            is_simulated=True,
            capabilities=frozenset({PRIVILEGED_SUCCESS, RESETTABLE}),
            supported_target_kinds=frozenset({"stack"}),
            docs=DOCS,
            environment_id=ENV_ID,
        )

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        layout = scene.metadata
        q = np.array([layout["start"]], np.float64)
        obs, _ = self._env.reset(
            options={
                "layout": {
                    "xy": np.array([[p[:2] for p in layout["positions"]]], np.float32),
                    "yaw": np.array([layout["yaws"]], np.float32),
                    "held": np.array([BLOCK_NAMES.index(layout["held"])]),
                    "target": np.array([BLOCK_NAMES.index(layout["target"])]),
                    "qpos": q.astype(np.float32),
                }
            }
        )
        self._q = q
        self._instruction = scene.instruction
        return self._observation(obs)

    def step(self, action: Action) -> StepResult:
        target = np.clip(action.data, LOW, HIGH)[None]
        refused = check_motion(target, self._joints) is not None
        # A refused target holds the arm where it was last sent, so the agent's arrival
        # check reports the shortfall. Measured from where the arm is, a move that meets a
        # block stops pressing once its command leads the arm by the step limit.
        if not refused:
            self._q, _ = solve(target)
        obs, _, _, _, info = self._env.step(
            torch.as_tensor(self._q, dtype=torch.float32, device=self._env.device)
        )
        success = bool(info["success"][0])
        return StepResult(
            observation=self._observation(obs),
            reward=float(success),
            terminated=success,
            termination_reason="success" if success else None,
            info={
                "success": success,
                "lifted": bool(info["lifted"][0]),
                "grasped": bool(info["grasped"][0]),
                "refused": refused,
            },
        )

    def pre_check(self, waypoints):
        """``check_motion`` from where the arm is; hand it to the agent as its ``pre_check``."""
        return check_motion(waypoints, self._joints)

    def close(self) -> None:
        self._env.close()

    def _observation(self, obs):
        pose = self._env.agent.tcp_pose
        qpos = self._env.agent.robot.get_qpos()[0].double().cpu().numpy()
        self._joints = qpos.copy()
        return Observation(
            images={name: obs["sensor_data"][name]["rgb"][0].cpu().numpy() for name in CAMERAS},
            state={
                "eef_state": read_pose(
                    pose.p[0].double().cpu().numpy(), pose.q[0].double().cpu().numpy(), qpos
                ),
                "joint_pos": qpos,
            },
            instruction=self._instruction,
        )
