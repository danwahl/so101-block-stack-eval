"""Score a VLA checkpoint on the block stack in sim.

    python scripts/run_vla.py pi05

Each policy is a LoRA adapter fine-tuned on the cell's oracle demonstrations
(drwahl/so101_block_stack_sim), published as drwahl/<name>_so101_block_stack_sim and
pinned here to the revision scored. It commands the six joint targets directly and executes
the number of actions of each chunk it scores best at before predicting the next. Each
chunk is read back off a smooth cubic through 13 of its steps, and a policy that supports
Real-Time Chunking (pi0, pi0.5, SmolVLA) stitches each chunk onto the unexecuted tail of
the one before; ``--no-smooth`` and ``--no-rtc`` turn these off. A VLA never ends a trial
itself, so every trial runs to the task's last step.

Needs the vla extra (`uv sync --extra vla`) and a GPU. Physics runs on the CPU;
`--sim-backend physx_cuda` runs it on the GPU, at about eight times the time.

Logs go to outputs/drwahl/<name>_so101_block_stack_sim/, and the leaderboard counts the
until-done task over the first 50 layouts, the default, as it does an agent's.
"""

import argparse

from huggingface_hub import snapshot_download
from inspect_robots import eval as ir_eval
from inspect_robots_so101.policy import LeRobotPolicy

from so101_block_stack_eval.embodiment import SimEmbodiment
from so101_block_stack_eval.spec import IMAGE_SIZE
from so101_block_stack_eval.task import stack

# name: (revision, lerobot policy type, actions executed per chunk)
POLICIES = {
    "pi05": ("e0f980f784146b7fa008edb19d773b31538155aa", "pi05", 10),
    "pi0": ("6bd755f49015017de405d9a9c08e5d6292255d78", "pi0", 10),
    "lawam": ("bab159946b7a89d777521a0d8da9e7918cb6e60f", "lawam", 10),
    "vla_jepa": ("f30c83bd3e0b986ec3f727c7fc55918ea815921a", "vla_jepa", 7),
    "smolvla": ("aff045fa3ec4ccdbaf54dcfabc7236b8dfa50dfb", "smolvla", 50),
}
# The lerobot policy types whose supports_rtc() is true.
RTC_TYPES = {"pi0", "pi05", "smolvla"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("policy", choices=POLICIES)
    parser.add_argument("--episodes", type=int, default=50, help="first N held-out layouts")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument(
        "--rtc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stitch chunks with Real-Time Chunking, where the policy supports it",
    )
    parser.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read each chunk back off a cubic through 13 of its steps",
    )
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--log-dir", help="defaults to outputs/drwahl/<repo>")
    args = parser.parse_args()

    revision, policy_type, chunk_size = POLICIES[args.policy]
    rtc = args.rtc and policy_type in RTC_TYPES
    repo = f"drwahl/{args.policy}_so101_block_stack_sim"
    checkpoint = snapshot_download(repo, revision=revision)
    policy = LeRobotPolicy(
        pretrained_path=f"{checkpoint}/adapter",
        policy_type=policy_type,
        cameras=("top", "wrist"),
        cam_height=IMAGE_SIZE,
        cam_width=IMAGE_SIZE,
        chunk_size=chunk_size,
        rtc=rtc,
        smooth=args.smooth,
        device="cuda",
    )
    embodiment = SimEmbodiment(until_done=True, control="joints", sim_backend=args.sim_backend)
    try:
        (log,) = ir_eval(
            stack(episodes=args.episodes, max_steps=args.max_steps, until_done=True),
            policy,
            embodiment,
            log_dir=args.log_dir or f"outputs/{repo}",
            policy_checkpoint=f"{repo}@{revision}",
        )
    finally:
        embodiment.close()
    print(log.results)


if __name__ == "__main__":
    main()
