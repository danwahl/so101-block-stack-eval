"""Closed-form inverse kinematics for the SO-101, batched over environments.

shoulder_pan is vertical, shoulder_lift / elbow_flex / wrist_flex are parallel and
horizontal, and wrist_roll is the tool approach axis. So the arm is a pan joint plus a
planar 3R plus a roll: fixing the tool tilt leaves a pan angle and a two-link cosine law,
and the roll is free to set the jaw yaw.
"""

from __future__ import annotations

import math

import torch

from .robot import BASE_POSE

BASE_P = torch.tensor(BASE_POSE.p, dtype=torch.float64)

# Link geometry in the pan frame (URDF joint origins).
PAN_ORIGIN = torch.tensor([0.045200, 0.000100, 0.069600], dtype=torch.float64)
LIFT_X, LIFT_Z = 0.030400, 0.054200
L1, A1 = 0.116030, math.atan2(-0.112600, 0.028000)  # shoulder_lift -> elbow_flex
L2, A2 = 0.135000, math.atan2(-0.005200, 0.134900)  # elbow_flex -> wrist_flex
L3 = 0.159258  # wrist_flex -> TCP along the roll axis

# The TCP sits 7.9 mm off the roll axis (the fixed jaw), at a fixed phase from the jaw
# axis, and the roll joint's zero is a quarter turn from it. Measured from FK; the
# read-back test in tests/test_block_stack.py pins them.
JAW_OFFSET = 0.007903
JAW_PHASE = 0.027605
ROLL_ZERO = -1.522120
AXIS_Y = -0.000177  # the roll axis' lateral offset from the pan axis

TILT_DOWN = math.pi / 2  # tool straight down


def tool_axes(pan_azimuth, jaw_yaw, tilt):
    """The gripper frame's three axes in world, for an arm aimed along ``pan_azimuth``.

    The arm's flex joints all turn about the horizontal square to its pan, so the tool
    axis stays in the vertical plane the arm reaches out along and ``tilt`` is its angle
    below the horizontal. The roll then turns the jaw about that axis by as much as
    ``jaw_yaw`` exceeds the pan, which is why a jaw axis is a world bearing with the tool
    pointing down and swings towards the vertical as the tool comes level.

    Returns ``(jaw, across, tool)``: the axis the jaws open along, its partner across the
    tool, and the approach axis, matching the gripper frame's own x, y and z.
    """
    cos_pan, sin_pan = torch.cos(pan_azimuth), torch.sin(pan_azimuth)
    zero, one = torch.zeros_like(cos_pan), torch.ones_like(cos_pan)
    radial = torch.stack([cos_pan, sin_pan, zero], dim=-1)
    sideways = torch.stack([-sin_pan, cos_pan, zero], dim=-1)
    up = torch.stack([zero, zero, one], dim=-1)

    cos_tilt, sin_tilt = torch.cos(tilt)[..., None], torch.sin(tilt)[..., None]
    tool = cos_tilt * radial - sin_tilt * up
    unrolled = sin_tilt * radial + cos_tilt * up
    roll = (jaw_yaw - pan_azimuth)[..., None]
    jaw = torch.cos(roll) * unrolled + torch.sin(roll) * sideways
    return jaw, torch.cross(tool, jaw, dim=-1), tool


def _jaw_offset(pan_azimuth, jaw_yaw, tilt):
    """The TCP's offset from the roll axis, in world.

    It sits across that axis at a fixed phase from the jaw, so it lies in the tool's own
    cross plane and turns with it: horizontal pointing straight down, vertical level.
    """
    jaw, across, _ = tool_axes(pan_azimuth, jaw_yaw, tilt)
    return JAW_OFFSET * (math.cos(JAW_PHASE) * jaw - math.sin(JAW_PHASE) * across)


def ik(tcp, jaw_yaw, tilt=TILT_DOWN):
    """Arm joint angles placing the TCP at world ``tcp`` with the tool at ``tilt``.

    ``tilt`` is the flex sum: ``TILT_DOWN`` points the tool straight down, less than that
    tilts it in from above, and ``jaw_yaw`` orients the jaw opening axis. The TCP's 7.9 mm
    offset from the roll axis is taken out, so ``tcp`` is where the tool actually lands.
    That offset lies across the roll axis, so it turns with the tool: horizontal pointing
    straight down, vertical with the tool level. Which way it lies depends on the pan the
    solve is about to produce, so the solve runs twice, the second time knowing it.

    Returns ``(q, reachable)`` with ``q`` shaped ``(..., 5)``. Angles where ``reachable``
    is false are meaningless (the two-link law has no solution there).
    """
    tcp = torch.as_tensor(tcp, dtype=torch.float64)
    kw = {"dtype": torch.float64, "device": tcp.device}
    jaw_yaw = torch.as_tensor(jaw_yaw, **kw).expand(tcp.shape[:-1])
    tilt = torch.as_tensor(tilt, **kw).expand(tcp.shape[:-1])

    reach = tcp - BASE_P.to(tcp.device) - PAN_ORIGIN.to(tcp.device)
    off = torch.zeros_like(reach)
    for _ in range(2):
        off = _jaw_offset(
            torch.atan2(reach[..., 1] - off[..., 1], reach[..., 0] - off[..., 0]), jaw_yaw, tilt
        )
    v = reach - off

    # Pan places the roll axis, which sits a fraction of a millimetre off that axis.
    rho = torch.hypot(v[..., 0], v[..., 1])
    radial = torch.sqrt(torch.clamp(rho * rho - AXIS_Y * AXIS_Y, min=0.0))
    pan = torch.atan2(torch.full_like(radial, AXIS_Y), radial) - torch.atan2(v[..., 1], v[..., 0])

    # Drop the wrist, whose direction the tilt already fixes, and solve the two links.
    u = radial - L3 * torch.cos(tilt) - LIFT_X
    w = v[..., 2] + L3 * torch.sin(tilt) - LIFT_Z
    cos_elbow = (u * u + w * w - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    reachable = (cos_elbow.abs() <= 1.0) & (rho > abs(AXIS_Y))
    elbow = torch.acos(torch.clamp(cos_elbow, -1.0, 1.0))

    lift = torch.atan2(-w, u) - torch.atan2(L2 * torch.sin(elbow), L1 + L2 * torch.cos(elbow)) - A1
    flex = tilt - lift - elbow + A2 - A1
    roll = torch.remainder(jaw_yaw + pan + ROLL_ZERO + math.pi, 2 * math.pi) - math.pi
    return torch.stack([pan, lift, elbow - A2 + A1, flex, roll], dim=-1), reachable
