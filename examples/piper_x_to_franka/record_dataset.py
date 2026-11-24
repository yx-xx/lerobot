#!/usr/bin/env python

"""Record LeRobot demonstrations with Piper-X teleoperation of Franka.

This is intentionally separate from ``lerobot-record`` because this setup uses
an extra Piper-X -> Franka action mapping processor.

Before starting, prepare ROS 2, CAN, and the Franka bridge as described in
``teleoperate.py``. A first local recording can be started with:

    python3 examples/piper_x_to_franka/record_dataset.py \
        --repo-id=local/piper-franka-pick \
        --root=data/piper-franka-pick \
        --task="Pick up the cube and place it in the bin" \
        --episodes=10 \
        --episode-time-s=30 \
        --reset-time-s=10

Use the right arrow to end the current episode early, the left arrow to
re-record the current episode, and Escape to stop after the current saved
episode. The default is local-only recording; pass ``--push-to-hub`` only
when the Hugging Face login and repository are ready.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.record import record_loop
from lerobot.robots.franka import FrankaRobot
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun

try:
    from .common import FPS, make_mapped_action_processor, make_piper_franka_configs
except ImportError:
    from common import FPS, make_mapped_action_processor, make_piper_franka_configs


DEFAULT_ROOT = Path("data/piper-franka-pick")
DEFAULT_REPO_ID = "local/piper-franka-pick"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Dataset id in owner/name format.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Local dataset directory.")
    parser.add_argument("--task", required=True, help="Single task description stored in every frame.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to record.")
    parser.add_argument("--episode-time-s", type=float, default=30.0)
    parser.add_argument("--reset-time-s", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--no-display", action="store_true", help="Disable Rerun visualization.")
    parser.add_argument("--resume", action="store_true", help="Append episodes to an existing local dataset.")
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the finished dataset to the Hugging Face Hub.",
    )
    parser.add_argument("--private", action="store_true", help="Use a private Hub dataset when uploading.")
    parser.add_argument("--writer-threads", type=int, default=4)
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.episode_time_s <= 0 or args.reset_time_s < 0:
        parser.error("episode and reset times must be non-negative, with episode time positive")
    if args.writer_threads < 0:
        parser.error("--writer-threads must be non-negative")
    return args


def make_dataset_features(robot, mapped_action_processor, robot_observation_processor, use_videos: bool):
    action_features = aggregate_pipeline_dataset_features(
        pipeline=mapped_action_processor,
        initial_features=create_initial_features(action=robot.action_features),
        use_videos=use_videos,
    )
    observation_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=robot.observation_features),
        use_videos=use_videos,
    )
    return combine_feature_dicts(action_features, observation_features)


def record_dataset(args: argparse.Namespace) -> LeRobotDataset:
    init_logging()
    logging.info("Recording Piper-X demonstrations for Franka: %s", args.task)

    teleop_config, robot_config = make_piper_franka_configs()
    teleop = make_teleoperator_from_config(teleop_config)
    robot = FrankaRobot(robot_config)
    mapped_action_processor = make_mapped_action_processor()
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = make_dataset_features(
        robot=robot,
        mapped_action_processor=mapped_action_processor,
        robot_observation_processor=robot_observation_processor,
        use_videos=True,
    )

    if args.resume:
        dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
        if robot.cameras:
            dataset.start_image_writer(
                num_processes=0,
                num_threads=args.writer_threads * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, args.fps, dataset_features)
    else:
        sanity_check_dataset_name(args.repo_id, policy_cfg=None)
        if args.root.exists():
            raise FileExistsError(
                f"Dataset root already exists: {args.root}. Use --resume to append to it, "
                "or choose another --root."
            )
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            root=args.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=args.writer_threads * len(robot.cameras),
        )

    if args.no_display:
        display_data = False
    else:
        display_data = True
        _init_rerun(session_name="piper_x_to_franka_record")

    listener = None
    try:
        robot.connect()
        teleop.connect()
        listener, events = init_keyboard_listener()

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < args.episodes and not events["stop_recording"]:
                episode_index = dataset.num_episodes
                log_say(f"Recording episode {episode_index}", blocking=False)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                    teleop_action_processor=mapped_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=args.episode_time_s,
                    single_task=args.task,
                    display_data=display_data,
                )

                if not events["stop_recording"] and (
                    recorded_episodes < args.episodes - 1 or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", blocking=False)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=args.fps,
                        teleop_action_processor=mapped_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=args.reset_time_s,
                        single_task=args.task,
                        display_data=display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", blocking=False)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                print(
                    f"Saved episode {dataset.num_episodes - 1} "
                    f"({recorded_episodes}/{args.episodes} in this run)."
                )
    finally:
        if listener is not None:
            listener.stop()
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()

    if args.push_to_hub:
        dataset.push_to_hub(private=args.private)

    return dataset


if __name__ == "__main__":
    record_dataset(parse_args())
