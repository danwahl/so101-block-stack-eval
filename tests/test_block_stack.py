import json

import httpx
import numpy as np
import pytest
import torch
from inspect_robots import eval as ir_eval
from inspect_robots.logging.sink import NullSink
from inspect_robots.types import Action
from inspect_robots_agent import LLMAgentPolicy, ToolError

from so101_block_stack_eval.embodiment import (
    HIGH,
    LABELS,
    MAX_STEP,
    TOP_CAM_TO_BASE,
    TOP_K,
    SimEmbodiment,
    check_motion,
    locate_points,
    read_pose,
    solve,
)
from so101_block_stack_eval.spec import HOME_QPOS, LIMITS
from so101_block_stack_eval.task import stack

# A pose over the middle of the spawn box, tool straight down, jaws open.
POSE = np.array([0.20, 0.0, 0.06, np.pi / 2, np.pi / 2, 1.0])


@pytest.fixture(scope="module")
def embodiment():
    emb = SimEmbodiment()
    yield emb
    emb.close()


@pytest.fixture(scope="module")
def layouts(tmp_path_factory):
    path = tmp_path_factory.mktemp("layouts") / "layouts.jsonl"
    path.write_text(
        json.dumps(
            {
                "index": 0,
                "held": 0,
                "target": 1,
                "prompt": "stack the red block on the green block",
                "positions": [[0.20, -0.06, 0.756], [0.24, 0.06, 0.756], [0.18, 0.08, 0.756]],
                "yaws": [0.3, -0.2, 0.0],
                "start": HOME_QPOS.tolist(),
            }
        )
        + "\n"
    )
    return str(path)


@pytest.fixture(scope="module")
def scene(layouts):
    return stack(layouts=layouts).scenes[0]


def _plan(emb, scene):
    """Grip point to block centre, close, lift, over the target, down a block, open."""
    held, target = (emb._env.blocks[name].pose for name in ("red", "green"))
    block = held.p[0].double().cpu().numpy() - [0.0, 0.0, 0.74]
    under = target.p[0].double().cpu().numpy() - [0.0, 0.0, 0.74]
    # A square block grips the same at any quarter turn; take the one nearest zero.
    yaw = (scene.metadata["yaws"][0] + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return [
        np.array(goal)
        for goal in (
            [*block[:2], 0.06, yaw, np.pi / 2, 1.0],
            [*block, yaw, np.pi / 2, 1.0],
            [*block, yaw, np.pi / 2, 0.0],
            [*block[:2], 0.07, yaw, np.pi / 2, 0.0],
            [*under[:2], 0.07, yaw, np.pi / 2, 0.0],
            [*under[:2], 0.047, yaw, np.pi / 2, 0.0],
            [*under[:2], 0.047, yaw, np.pi / 2, 1.0],
        )
    ]


def _move(emb, start, goal):
    """Straight-line steps from ``start`` to ``goal`` at the declared per-step limits."""
    steps = max(1, int(np.ceil(np.max(np.abs(goal - start) / np.array(MAX_STEP)))))
    result = None
    for f in np.linspace(1.0 / steps, 1.0, steps):
        result = emb.step(Action(data=start + f * (goal - start)))
    for _ in range(5):
        result = emb.step(Action(data=goal))
    return result


def test_the_task_reads_the_held_out_layouts_into_scenes(scene):
    assert scene.instruction == "stack the red block on the green block"
    assert scene.target.spec == {"held": "red", "target": "green"}


def test_a_trial_opens_at_its_layouts_start(embodiment, scene):
    obs = embodiment.reset(scene)
    assert set(obs.images) == {"top", "wrist"}
    assert np.abs(obs.state["joint_pos"] - HOME_QPOS).max() < 1e-6


def test_the_packaged_layouts_open_in_reach_near_the_rest_roll():
    """Every start is inside the joints' travel, its wrist roll within a quarter turn
    centred on the rest pose's, and no two layouts open alike."""
    scenes = stack().scenes
    assert len(scenes) == 150
    starts = np.array([scene.metadata["start"] for scene in scenes])
    assert ((starts >= LIMITS[:, 0]) & (starts <= LIMITS[:, 1])).all()
    assert (np.abs(starts[:, 4] - HOME_QPOS[4]) <= np.pi / 4 + 1e-6).all()
    assert len(np.unique(starts.round(4), axis=0)) == len(starts)


def test_tilted_commands_land_where_they_say(embodiment, scene):
    """Grip point, yaw and pitch, read back off the sim's own TCP frame."""
    embodiment.reset(scene)
    rng = np.random.default_rng(0)
    targets = np.stack(
        [
            rng.uniform(0.14, 0.26, 64),
            rng.uniform(-0.12, 0.12, 64),
            rng.uniform(0.02, 0.12, 64),
            rng.uniform(-1.4, 1.4, 64),
            rng.uniform(np.pi / 2 - 1.2, np.pi / 2 + 0.4, 64),
            np.ones(64),
        ],
        -1,
    )
    q, ok = solve(targets)
    assert ok.sum() > 20
    robot, tcp = embodiment._env.agent.robot, embodiment._env.agent
    for target, joints in zip(targets[ok], q[ok], strict=True):
        robot.set_qpos(torch.as_tensor(joints[None], dtype=torch.float32))
        pose = tcp.tcp_pose
        got = read_pose(pose.p[0].double().cpu().numpy(), pose.q[0].double().cpu().numpy(), joints)
        assert np.abs(got[:3] - target[:3]).max() < 1e-3
        assert np.abs(got[3:5] - target[3:5]).max() < 1e-4


def test_the_locate_points_calibration_is_the_sim_camera(embodiment):
    """locate_points' calibration projects points as the sim's camera does."""
    params = embodiment._env.scene.sensors["top"].get_params()
    world_to_cam = params["extrinsic_cv"][0].double().cpu().numpy()
    k = params["intrinsic_cv"][0].double().cpu().numpy()
    base = embodiment._env.agent.robot.pose.p[0].double().cpu().numpy()
    base_to_cam = np.linalg.inv(TOP_CAM_TO_BASE)
    rng = np.random.default_rng(0)
    for point in np.stack(
        [rng.uniform(0.1, 0.4, 32), rng.uniform(-0.2, 0.2, 32), rng.uniform(0.0, 0.1, 32)], -1
    ):
        sim = k @ (world_to_cam[:, :3] @ (point + base) + world_to_cam[:, 3])
        stated = TOP_K @ (base_to_cam[:3, :3] @ point + base_to_cam[:3, 3])
        assert np.abs(sim[:2] / sim[2] - stated[:2] / stated[2]).max() < 1.0


def test_locate_points_finds_a_block_from_its_top_face(embodiment, scene):
    """The pixel the sim's camera puts a block's top-face centre at locates the block."""
    embodiment.reset(scene)
    params = embodiment._env.scene.sensors["top"].get_params()
    world_to_cam = params["extrinsic_cv"][0].double().cpu().numpy()
    k = params["intrinsic_cv"][0].double().cpu().numpy()
    base = embodiment._env.agent.robot.pose.p[0].double().cpu().numpy()
    centres = [
        embodiment._env.blocks[name].pose.p[0].double().cpu().numpy() - base
        for name in ("red", "green")
    ]
    pixels = []
    for centre in centres:
        top = k @ (world_to_cam[:, :3] @ (centre + base + [0.0, 0.0, 0.015]) + world_to_cam[:, 3])
        pixels.append([float(top[0] / top[2]), float(top[1] / top[2])])
    lines = locate_points({"points": pixels, "height": 0.03}, None).splitlines()
    for line, centre in zip(lines, centres, strict=True):
        x, y, z = (float(part.split()[1]) for part in line.split("-> ")[1].split(", "))
        assert abs(x - centre[0]) < 1e-3 and abs(y - centre[1]) < 1e-3 and z == 0.03
    for bad in (
        {"points": [[600, 10]], "height": 0.0},
        {"points": [1, 2], "height": 0.0},
        {"points": [[1, 2]], "height": 1},
        {"points": [[1, 2]]},
    ):
        with pytest.raises(ToolError):
            locate_points(bad, None)


def test_moves_out_of_reach_are_refused_with_a_reason():
    nearby = POSE + np.array([0.005, 0.0, -0.005, 0.05, -0.05, 0.0])
    assert check_motion(np.stack([POSE, nearby])) is None
    reason = check_motion(np.stack([POSE, HIGH]))
    assert reason is not None and "out of reach at the target" in reason
    # Straight down, the grip cannot rise to 0.14: the path fails at about 0.10, but the
    # reason names the target.
    high = np.linspace(POSE, POSE + np.array([0.0, 0.0, 0.08, 0.0, 0.0, 0.0]), 16)
    ok = solve(high)[1]
    assert ok[0] and not ok[-2:].any()
    reason = check_motion(high)
    assert reason is not None and "out of reach at the target" in reason


def test_a_lunge_is_refused():
    """A target that solves, but a long way from where the arm was last sent."""
    start, _ = solve(POSE[None])
    far = POSE + np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    assert solve(far[None])[1].all()
    assert check_motion(far[None]) is None
    reason = check_motion(far[None], start[0])
    assert reason is not None and "degrees in one step" in reason


def test_moves_are_checked_from_where_the_arm_stalled(embodiment, scene):
    """An arm left short of its last command, as a block would leave it, can still move."""
    here = embodiment.reset(scene).state["eef_state"]
    embodiment._q, _ = solve((here + np.array([-0.06, 0.0, -0.06, 0.0, 0.0, 0.0]))[None])
    nudge = here + np.array([-0.004, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert embodiment.pre_check(nudge[None]) is None
    assert not embodiment.step(Action(data=nudge)).info["refused"]


def test_a_refused_step_holds_the_last_target(embodiment, scene):
    embodiment.reset(scene)
    held = embodiment._q.copy()
    result = embodiment.step(Action(data=HIGH))
    assert result.info["refused"]
    assert (embodiment._q == held).all()


class _Records(NullSink):
    def __init__(self):
        self.records = []

    def on_trial_end(self, record):
        self.records.append(record)


def test_an_agent_stacks_through_eval(embodiment, scene, layouts, tmp_path):
    """The agent's own tools and the framework's rollout, with the model scripted."""
    embodiment.reset(scene)
    turns = [
        (
            "move_to",
            {
                "targets": dict(zip(LABELS, map(float, goal), strict=True)),
                "note": "following the plan",
            },
        )
        for goal in _plan(embodiment, scene)
    ]
    turns.append(("done", {"summary": "stacked", "hindsight": "none"}))

    def reply(request):
        name, arguments = turns.pop(0) if len(turns) > 1 else turns[0]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call_{name}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(arguments)},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    policy = LLMAgentPolicy(
        model="test/model",
        base_url="http://llm.test/v1",
        transport=httpx.MockTransport(reply),
        env={},
        pre_check=embodiment.pre_check,
    )
    sink = _Records()
    (log,) = ir_eval(
        stack(layouts=layouts), policy, embodiment, log_dir=str(tmp_path), sinks=[sink]
    )
    assert log.status == "success"
    (record,) = sink.records
    assert record.termination_reason == "success"
    assert log.results.metrics == {"success_at_end": 1.0, "lifted": 1.0}
