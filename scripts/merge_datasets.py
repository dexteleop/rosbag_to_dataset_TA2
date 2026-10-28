#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Merge multiple LeRobot datasets into a single dataset.

Before running this script:
1. Place all source datasets under a common root directory
2. Prepare the output directory for the merged dataset
3. Run with appropriate root and output parameters

Usage example:
    python scripts/merge_datasets.py --root ./data/source_datasets --output ./output/merged_dataset
"""
import argparse
import contextlib
import json
import os
import shutil
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Constants
DEFAULT_CHUNK_SIZE = 1000
VECTOR_KEYS = ("action", "observation.state")

def load_jsonl(file_path):
    """Load data from a JSONL file
    Args:
        file_path (str): Path to the JSONL file
    Returns:
        list: List containing JSON objects from each line
    """
    data = []

    # Special handling for episodes_stats.jsonl
    if "episodes_stats.jsonl" in file_path:
        try:
            # Try to load the entire file as a JSON array
            with open(file_path) as f:
                content = f.read()
                # Check if the content starts with '[' and ends with ']'
                if content.strip().startswith("[") and content.strip().endswith("]"):
                    return json.loads(content)
                else:
                    # Try to add brackets and parse
                    try:
                        return json.loads("[" + content + "]")
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Error loading {file_path} as JSON array: {e}")

        # Fall back to line-by-line parsing
        try:
            with open(file_path) as f:
                for line in f:
                    if line.strip():
                        with contextlib.suppress(json.JSONDecodeError):
                            data.append(json.loads(line))
        except Exception as e:
            print(f"Error loading {file_path} line by line: {e}")
    else:
        # Standard JSONL parsing for other files
        with open(file_path) as f:
            for line in f:
                if line.strip():
                    with contextlib.suppress(json.JSONDecodeError):
                        data.append(json.loads(line))

    return data

def save_jsonl(data, file_path):
    """Save data in JSONL format
    Args:
        data (list): List of JSON objects to save
        file_path (str): Path to the output file
    """
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def copy_videos(source_folders, output_folder, episode_mapping, chunks_size=DEFAULT_CHUNK_SIZE):
    """Copy video files from source folders to output folder, maintaining correct indices and structure.

    Args:
        source_folders (list): List of source dataset folder paths
        output_folder (str): Output folder path
        episode_mapping (list): List of tuples containing (old_folder, old_index, new_index)
        chunks_size (int): Number of episodes per chunk
    """
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    video_path_template = info["video_path"]

    video_keys = []
    for feature_name, feature_info in info["features"].items():
        if feature_info.get("dtype") == "video":
            video_keys.append(feature_name)

    print(f"Found video keys: {video_keys}")

    for old_folder, old_index, new_index in episode_mapping:
        episode_chunk = old_index // chunks_size
        new_episode_chunk = new_index // chunks_size

        for video_key in video_keys:
            # Only accept paths that match this episode's index — falling back to
            # another episode's video silently corrupts the merged dataset
            source_patterns = [
                os.path.join(
                    old_folder,
                    video_path_template.format(
                        episode_chunk=episode_chunk, video_key=video_key, episode_index=old_index
                    ),
                ),
                os.path.join(
                    old_folder,
                    f"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{old_index}.mp4"
                ),
            ]

            source_video_path = None
            for pattern in source_patterns:
                if os.path.exists(pattern):
                    source_video_path = pattern
                    break

            if source_video_path:
                dest_video_path = os.path.join(
                    output_folder,
                    video_path_template.format(
                        episode_chunk=new_episode_chunk, video_key=video_key, episode_index=new_index
                    ),
                )

                os.makedirs(os.path.dirname(dest_video_path), exist_ok=True)

                print(f"Copying video: {source_video_path} -> {dest_video_path}")
                shutil.copy2(source_video_path, dest_video_path)
            else:
                print(f"ERROR: Video file not found for {video_key}, episode {old_index} in {old_folder}; "
                      f"the merged dataset will be missing this video")


def vector_columns(table):
    """Return {column: ndarray} for the vector-valued feature columns present in the table."""
    out = {}
    for key in VECTOR_KEYS:
        if key in table.column_names:
            out[key] = np.stack(table[key].to_numpy(zero_copy_only=False))
    return out


def stats_entry(values):
    """Per-episode statistics in the format LeRobot writes to episodes_stats.jsonl."""
    return {
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "count": [int(values.shape[0])],
    }


def build_episode_stats(table):
    """Recompute statistics for every numeric feature of one episode.

    Renumbering rewrites episode_index and index, and merging can remap task_index, so the
    source statistics of those columns cannot be reused. Vector columns are recomputed from
    the merged data; scalar columns are recomputed the same way.
    """
    stats = {}
    for key, values in vector_columns(table).items():
        stats[key] = stats_entry(values.astype(np.float64))
    for key in table.column_names:
        if key in VECTOR_KEYS:
            continue
        values = table[key].to_numpy(zero_copy_only=False)
        if values.dtype.kind in "iufb":
            stats[key] = stats_entry(values.astype(np.float64).reshape(-1, 1))
    return stats


def copy_data_files(
    source_folders,
    output_folder,
    episode_mapping,
    episode_to_frame_index=None,
    folder_task_mapping=None,
    chunks_size=DEFAULT_CHUNK_SIZE,
):
    """Copy and process parquet data files, updating episode, frame, and task indices.

    Args:
        source_folders (list): List of source dataset folder paths
        output_folder (str): Output folder path
        episode_mapping (list): List of tuples containing (old_folder, old_index, new_index)
        episode_to_frame_index (dict, optional): Mapping of each new episode index to its starting frame index
        folder_task_mapping (dict, optional): Mapping of task_index for each folder
        chunks_size (int): Number of episodes per chunk
    """
    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    data_path_template = info["data_path"]
    episode_stats = {}

    for old_folder, old_index, new_index in episode_mapping:
        episode_chunk = old_index // chunks_size
        source_path = os.path.join(
            old_folder,
            data_path_template.format(episode_chunk=episode_chunk, episode_index=old_index),
        )

        try:
            table = pq.read_table(source_path)
            num_rows = table.num_rows

            position = table.schema.get_field_index("episode_index")
            if position != -1:
                print(f"Update episode_index from {table['episode_index'][0].as_py()} to {new_index}")
                table = table.set_column(position, table.schema.field(position),
                                         pa.array(np.full(num_rows, new_index, dtype=np.int64),
                                                  type=table.schema.field(position).type))

            position = table.schema.get_field_index("index")
            if position != -1:
                if episode_to_frame_index and new_index in episode_to_frame_index:
                    first_index = episode_to_frame_index[new_index]
                    print(f"Update index column, start value: {first_index} (using global cumulative frame count)")
                else:
                    first_index = new_index * num_rows
                    print(f"Update index column, start value: {first_index} (using episode index multiplied by length)")

                table = table.set_column(position, table.schema.field(position),
                                         pa.array(np.arange(first_index, first_index + num_rows, dtype=np.int64),
                                                  type=table.schema.field(position).type))

            position = table.schema.get_field_index("task_index")
            if position != -1 and folder_task_mapping and old_folder in folder_task_mapping:
                current_task_index = table["task_index"][0].as_py()

                if current_task_index in folder_task_mapping[old_folder]:
                    new_task_index = folder_task_mapping[old_folder][current_task_index]
                    print(f"Update task_index from {current_task_index} to {new_task_index}")
                    table = table.set_column(position, table.schema.field(position),
                                             pa.array(np.full(num_rows, new_task_index, dtype=np.int64),
                                                      type=table.schema.field(position).type))
                else:
                    print(f"Warning: No mapping found for task_index {current_task_index}")

            chunk_index = new_index // chunks_size

            chunk_dir = os.path.join(output_folder, "data", f"chunk-{chunk_index:03d}")
            os.makedirs(chunk_dir, exist_ok=True)
            dest_path = os.path.join(chunk_dir, f"episode_{new_index:06d}.parquet")

            pq.write_table(table, dest_path, compression="snappy")
            episode_stats[new_index] = build_episode_stats(table)
            print(f"Processed and saved: {dest_path}")

        except Exception as e:
            error_msg = f"Processing {source_path} failed: {e}"
            print(error_msg)
            traceback.print_exc()

    return episode_stats


def merge_datasets(source_folders, output_folder):
    """Merge multiple dataset folders into one, handling indices, dimensions, and metadata.

    Args:
        source_folders (list): List of source dataset folder paths
        output_folder (str): Output folder path
    """
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "meta"), exist_ok=True)

    all_episodes = []
    all_tasks = []
    all_unique_tasks = []

    total_frames = 0
    total_episodes = 0
    total_videos = 0

    chunks_size = DEFAULT_CHUNK_SIZE

    episode_mapping = []
    folder_dimensions = {}

    cumulative_frame_count = 0
    episode_to_frame_index = {}
    task_desc_to_new_index = {}
    folder_task_mapping = {}

    for folder in source_folders:
        try:
            folder_info_path = os.path.join(folder, "meta", "info.json")
            if not os.path.exists(folder_info_path):
                print(f"Warning: info.json not found in {folder}, skipping")
                continue
            with open(folder_info_path) as f:
                folder_info = json.load(f)

            if "total_videos" in folder_info:
                folder_videos = folder_info["total_videos"]
                total_videos += folder_videos
                print(f"Read video count from {folder}'s info.json: {folder_videos}")

            folder_dim = folder_info["features"]["observation.state"]["shape"]
            folder_dimensions[folder] = folder_dim

            episodes_path = os.path.join(folder, "meta", "episodes.jsonl")
            if not os.path.exists(episodes_path):
                print(f"Warning: Episodes file not found in {folder}, skipping")
                continue
            episodes = load_jsonl(episodes_path)

            tasks_path = os.path.join(folder, "meta", "tasks.jsonl")
            folder_tasks = []
            if os.path.exists(tasks_path):
                folder_tasks = load_jsonl(tasks_path)

            folder_task_mapping[folder] = {}

            for task in folder_tasks:
                task_desc = task["task"]
                old_index = task["task_index"]

                if task_desc not in task_desc_to_new_index:
                    new_index = len(all_unique_tasks)
                    task_desc_to_new_index[task_desc] = new_index
                    all_unique_tasks.append({"task_index": new_index, "task": task_desc})

                folder_task_mapping[folder][old_index] = task_desc_to_new_index[task_desc]

            for episode in episodes:
                old_index = episode["episode_index"]
                new_index = total_episodes

                episode["episode_index"] = new_index
                all_episodes.append(episode)

                episode_mapping.append((folder, old_index, new_index))

                total_episodes += 1
                total_frames += episode["length"]

                episode_to_frame_index[new_index] = cumulative_frame_count
                cumulative_frame_count += episode["length"]

            all_tasks = all_unique_tasks

        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue

    print(f"Processed {total_episodes} episodes from {len(source_folders)} folders")

    save_jsonl(all_episodes, os.path.join(output_folder, "meta", "episodes.jsonl"))
    save_jsonl(all_tasks, os.path.join(output_folder, "meta", "tasks.jsonl"))

    info_path = os.path.join(source_folders[0], "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(all_tasks)
    info["total_chunks"] = (total_episodes + info["chunks_size"] - 1) // info["chunks_size"]

    info["splits"] = {"train": f"0:{total_episodes}"}

    info["total_videos"] = total_videos
    print(f"Update total videos to: {total_videos}")

    with open(os.path.join(output_folder, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    copy_videos(
        source_folders,
        output_folder,
        episode_mapping,
        chunks_size,
    )
    episode_stats = copy_data_files(
        source_folders,
        output_folder,
        episode_mapping,
        episode_to_frame_index,
        folder_task_mapping,
        chunks_size,
    )

    # Statistics are recomputed from the merged parquet files, because episode_index, index
    # and task_index are all rewritten during the merge and the source stats no longer apply.
    missing = [ep for ep in range(total_episodes) if ep not in episode_stats]
    if missing:
        print(f"Warning: no statistics computed for {len(missing)} episode(s): {missing[:10]}")
    merged_stats = [{"episode_index": ep, "stats": episode_stats[ep]} for ep in sorted(episode_stats)]
    save_jsonl(merged_stats, os.path.join(output_folder, "meta", "episodes_stats.jsonl"))

    print(f"Merged {total_episodes} episodes with {total_frames} frames into {output_folder}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Merge datasets from multiple sources.")

    # Add arguments
    parser.add_argument("--root", type=str,
                        default='./data/source_datasets',
                        help="Root of source folder paths")
    parser.add_argument("--output", type=str,
                        default='./output/merged_dataset',
                        help="Output folder path")

    # Parse arguments
    args = parser.parse_args()

    # Use parsed arguments
    root=Path(args.root)
    sources = [str(p) for p in root.iterdir() if p.is_dir()]
    merge_datasets(sources, args.output)
