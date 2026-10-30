#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.
#
# The on-disk layout produced here follows LeRobot's own converter,
# lerobot/datasets/v30/convert_dataset_v21_to_v30.py (Apache-2.0, The
# HuggingFace Inc. team), from lerobot 0.4.4.

"""
Convert a local LeRobot dataset from format v2.1 to v3.0.

v3.0 stores many episodes per file instead of one file per episode, and keeps
episode boundaries in metadata:

    v2.1                                        v3.0
    data/chunk-000/episode_000000.parquet       data/chunk-000/file-000.parquet
    videos/chunk-000/CAM/episode_000000.mp4     videos/CAM/chunk-000/file-000.mp4
    meta/episodes.jsonl                         meta/episodes/chunk-000/file-000.parquet
    meta/episodes_stats.jsonl                   (per-episode stats/... columns above)
                                                meta/stats.json (aggregated)
    meta/tasks.jsonl                            meta/tasks.parquet
    meta/info.json                              meta/info.json (codebase_version v3.0)

This is a standalone version of LeRobot's converter. It needs no particular
lerobot release, so it runs in the same environment as the rosbag converters
(lerobot 0.3.3), and works on a local directory only: nothing is downloaded
from or pushed to the Hugging Face Hub.

It differs from the upstream script in three ways:

  * Data files keep the source parquet schema (pyarrow, not a pandas round
    trip), so fixed-size list columns and the `huggingface` schema metadata
    survive.
  * An episode that starts a new data file is recorded in that file. Upstream
    records it in the previous file whenever a file fills up, which points
    the reader at the wrong file for that episode.
  * The result is checked before it is kept: every episode's rows, frame
    count and video span are compared with the source (see --no-verify).

Videos are not re-encoded: per-episode MP4s are concatenated with stream copy,
so every frame keeps the exact bytes it had. Because the rosbag converters
encode every frame as a keyframe, any frame of the concatenated file can still
be decoded on its own.

Usage
-----
    conda activate rosbag2lerobot
    unset PYTHONPATH

    # Writes /data/my_dataset_v30, leaving /data/my_dataset untouched
    python scripts/convert_lerobot_v21_to_v30.py --input /data/my_dataset

    python scripts/convert_lerobot_v21_to_v30.py \
        --input /data/my_dataset --output /data/v30/my_dataset

The output directory must not exist yet; the conversion is written to a
temporary directory next to it and moved into place only once it verified.
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SOURCE_VERSION = "v2.1"
TARGET_VERSION = "v3.0"

# Defaults of lerobot 0.4.4 (lerobot/datasets/utils.py)
DEFAULT_CHUNK_SIZE = 1000          # files per chunk directory
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DATA_PATH = "data/" + CHUNK_FILE_PATTERN + ".parquet"
VIDEO_PATH = "videos/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
EPISODES_PATH = "meta/episodes/" + CHUNK_FILE_PATTERN + ".parquet"
TASKS_PATH = "meta/tasks.parquet"
STATS_PATH = "meta/stats.json"
INFO_PATH = "meta/info.json"

LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"

# A converted video span may differ from length/fps by container rounding of
# the last frame's duration; anything beyond one frame means frames were lost.
VIDEO_SPAN_TOLERANCE_FRAMES = 1.0

log = logging.getLogger("convert_lerobot_v21_to_v30")


class ConversionError(Exception):
    pass


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ConversionError(f"missing {path}")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def next_chunk_file(chunk_index: int, file_index: int, chunk_size: int) -> tuple[int, int]:
    if file_index == chunk_size - 1:
        return chunk_index + 1, 0
    return chunk_index, file_index + 1


def parquet_size_in_mb(path: Path) -> float:
    """Uncompressed size, the measure LeRobot uses to cap data files."""
    meta = pq.read_metadata(path)
    total = 0
    for rg in range(meta.num_row_groups):
        row_group = meta.row_group(rg)
        for col in range(row_group.num_columns):
            total += row_group.column(col).total_uncompressed_size
    return total / 1024 ** 2


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in d.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, name))
        else:
            out[name] = value
    return out


def video_duration_in_s(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        return float(container.duration / av.time_base)


def concatenate_videos(paths: list[Path], output: Path) -> None:
    """Join MP4s by stream copy (no re-encode), as lerobot 0.4.4 does."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        listing = Path(tmp) / "inputs.ffconcat"
        listing.write_text("ffconcat version 1.0\n" + "".join(
            f"file '{p.resolve()}'\n" for p in paths))
        partial = Path(tmp) / output.name
        source = av.open(str(listing), mode="r", format="concat", options={"safe": "0"})
        target = av.open(str(partial), mode="w", options={"movflags": "faststart"})
        try:
            streams = {}
            for stream in source.streams:
                if stream.type in ("video", "audio", "subtitle"):
                    out_stream = target.add_stream_from_template(template=stream, opaque=True)
                    out_stream.time_base = stream.time_base
                    streams[stream.index] = out_stream
            for packet in source.demux():
                if packet.stream.index not in streams or packet.dts is None:
                    continue
                packet.stream = streams[packet.stream.index]
                target.mux(packet)
        finally:
            source.close()
            target.close()
        shutil.move(str(partial), str(output))


def aggregate_stats(per_episode: list[dict]) -> dict:
    """Dataset-level stats from per-episode stats (lerobot's parallel algorithm).

    The mean is count-weighted; the variance combines each episode's variance
    with its mean's offset from the total mean, so the result is exactly the
    stats of all frames together. min/max are elementwise.
    """
    features = sorted({k for stats in per_episode for k in stats})
    out = {}
    for feature in features:
        items = [s[feature] for s in per_episode if feature in s]
        means = np.stack([np.asarray(s["mean"], dtype=np.float64) for s in items])
        variances = np.stack([np.asarray(s["std"], dtype=np.float64) ** 2 for s in items])
        counts = np.stack([np.asarray(s["count"], dtype=np.float64) for s in items])
        total = counts.sum(axis=0)
        weights = counts
        while weights.ndim < means.ndim:
            weights = np.expand_dims(weights, axis=-1)
        mean = (means * weights).sum(axis=0) / total
        variance = ((variances + (means - mean) ** 2) * weights).sum(axis=0) / total
        out[feature] = {
            "min": np.min(np.stack([np.asarray(s["min"]) for s in items]), axis=0).tolist(),
            "max": np.max(np.stack([np.asarray(s["max"]) for s in items]), axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": total.astype(np.int64).tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# the source dataset
# ---------------------------------------------------------------------------

class SourceDataset:
    """A v2.1 dataset, validated before anything is written."""

    def __init__(self, root: Path):
        self.root = root
        info_path = root / INFO_PATH
        if not info_path.is_file():
            raise ConversionError(f"{root} is not a LeRobot dataset (no {INFO_PATH})")
        self.info = read_json(info_path)
        version = self.info.get("codebase_version")
        if version != SOURCE_VERSION:
            raise ConversionError(f"{root} is codebase_version {version!r}; this script converts "
                                  f"{SOURCE_VERSION} only")

        self.fps = int(self.info["fps"])
        self.chunk_size = int(self.info.get("chunks_size", DEFAULT_CHUNK_SIZE))
        features = self.info["features"]
        self.video_keys = sorted(k for k, f in features.items() if f["dtype"] == "video")
        self.image_keys = sorted(k for k, f in features.items() if f["dtype"] == "image")

        self.episodes = sorted(read_jsonl(root / LEGACY_EPISODES_PATH),
                               key=lambda e: e["episode_index"])
        indices = [e["episode_index"] for e in self.episodes]
        if indices != list(range(len(indices))):
            raise ConversionError("episode_index in episodes.jsonl is not contiguous 0..N-1")
        if not self.episodes:
            raise ConversionError("the dataset has no episodes")

        stats = {s["episode_index"]: s["stats"] for s in read_jsonl(root / LEGACY_EPISODES_STATS_PATH)}
        missing = [i for i in indices if i not in stats]
        if missing:
            raise ConversionError(f"episodes_stats.jsonl has no stats for episodes {missing[:10]}")
        self.episode_stats = [stats[i] for i in indices]

        tasks = sorted(read_jsonl(root / LEGACY_TASKS_PATH), key=lambda t: t["task_index"])
        self.tasks = {t["task_index"]: t["task"] for t in tasks}

        data_template = self.info["data_path"]
        video_template = self.info.get("video_path")
        self.data_files, self.video_files = [], {k: [] for k in self.video_keys}
        problems = []
        for episode in self.episodes:
            i = episode["episode_index"]
            chunk = i // self.chunk_size
            path = root / data_template.format(episode_chunk=chunk, episode_index=i)
            if not path.is_file():
                problems.append(f"missing {path.relative_to(root)}")
            self.data_files.append(path)
            for key in self.video_keys:
                path = root / video_template.format(episode_chunk=chunk, video_key=key, episode_index=i)
                if not path.is_file():
                    problems.append(f"missing {path.relative_to(root)}")
                self.video_files[key].append(path)
        if problems:
            raise ConversionError(f"{len(problems)} file(s) missing, e.g. {problems[:3]}")

        declared = set(features)
        extra = set(pq.read_schema(self.data_files[0]).names) - declared
        if extra:
            # Upstream fails later with an opaque schema-cast error on these.
            raise ConversionError(f"data columns not declared in info.json features: {sorted(extra)}")


# ---------------------------------------------------------------------------
# conversion steps (each writes under `out`)
# ---------------------------------------------------------------------------

def convert_info(src: SourceDataset, out: Path, data_mb: int, video_mb: int) -> None:
    info = json.loads(json.dumps(src.info))
    info["codebase_version"] = TARGET_VERSION
    info.pop("total_chunks", None)
    info.pop("total_videos", None)
    info["data_files_size_in_mb"] = data_mb
    info["video_files_size_in_mb"] = video_mb
    info["data_path"] = DATA_PATH
    info["video_path"] = VIDEO_PATH if src.info.get("video_path") is not None else None
    info["fps"] = src.fps
    for key, feature in info["features"].items():
        if feature["dtype"] != "video":
            feature["fps"] = src.fps
    write_json(info, out / INFO_PATH)


def convert_tasks(src: SourceDataset, out: Path) -> None:
    frame = pd.DataFrame({"task_index": list(src.tasks)}, index=list(src.tasks.values()))
    path = out / TASKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def convert_data(src: SourceDataset, out: Path, data_mb: int) -> list[dict]:
    """Concatenate per-episode parquet files into files of about data_mb each."""
    metadata = []
    chunk, file, size_mb, row = 0, 0, 0.0, 0
    batch: list[Path] = []
    schema = pq.read_schema(src.data_files[0])

    def flush():
        if not batch:
            return
        table = pa.concat_tables([pq.read_table(p, schema=schema) for p in batch])
        path = out / DATA_PATH.format(chunk_index=chunk, file_index=file)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

    for episode, path in zip(src.episodes, src.data_files):
        episode_mb = parquet_size_in_mb(path)
        rows = pq.read_metadata(path).num_rows
        if rows != episode["length"]:
            raise ConversionError(f"{path.name} has {rows} rows, episodes.jsonl says "
                                  f"{episode['length']}")
        if batch and size_mb + episode_mb >= data_mb:
            flush()
            chunk, file = next_chunk_file(chunk, file, DEFAULT_CHUNK_SIZE)
            batch, size_mb = [], 0.0
        # Recorded after any roll-over, so it names the file the episode is in.
        metadata.append({
            "episode_index": episode["episode_index"],
            "data/chunk_index": chunk,
            "data/file_index": file,
            "dataset_from_index": row,
            "dataset_to_index": row + rows,
        })
        batch.append(path)
        size_mb += episode_mb
        row += rows
    flush()
    return metadata


def convert_videos_of_camera(src: SourceDataset, out: Path, key: str, video_mb: int) -> list[dict]:
    metadata = []
    chunk, file, size_mb, offset_s = 0, 0, 0.0, 0.0
    batch: list[Path] = []

    def flush():
        concatenate_videos(batch, out / VIDEO_PATH.format(video_key=key, chunk_index=chunk,
                                                          file_index=file))

    for episode, path in zip(src.episodes, src.video_files[key]):
        episode_mb = path.stat().st_size / 1024 ** 2
        duration = video_duration_in_s(path)
        if batch and size_mb + episode_mb >= video_mb:
            flush()
            chunk, file = next_chunk_file(chunk, file, DEFAULT_CHUNK_SIZE)
            batch, size_mb, offset_s = [], 0.0, 0.0
        metadata.append({
            f"videos/{key}/chunk_index": chunk,
            f"videos/{key}/file_index": file,
            f"videos/{key}/from_timestamp": offset_s,
            f"videos/{key}/to_timestamp": offset_s + duration,
        })
        batch.append(path)
        size_mb += episode_mb
        offset_s += duration
    if batch:
        flush()
    return metadata


def convert_episodes(src: SourceDataset, out: Path, data_meta: list[dict],
                     video_meta: dict[str, list[dict]]) -> None:
    rows = []
    for i, episode in enumerate(src.episodes):
        row = {**data_meta[i]}
        for key in src.video_keys:
            row.update(video_meta[key][i])
        row["tasks"] = list(episode["tasks"])
        row["length"] = episode["length"]
        row.update(flatten({"stats": src.episode_stats[i]}))
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
        rows.append(row)

    # Written through `datasets`, as upstream does, so lerobot's loader sees
    # the same schema it would for a dataset it converted itself.
    from datasets import Dataset
    episodes = Dataset.from_list(rows)
    path = out / EPISODES_PATH.format(chunk_index=0, file_index=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(str(path))

    write_json(aggregate_stats(src.episode_stats), out / STATS_PATH)


def copy_extra_files(src: SourceDataset, out: Path) -> None:
    """Keep top-level files that are not part of the format (e.g. the
    converter's review_report.json); the format's own files were rewritten."""
    for path in sorted(src.root.iterdir()):
        if path.is_file() and not path.name.startswith("."):
            shutil.copy2(path, out / path.name)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def verify(src: SourceDataset, out: Path) -> list[str]:
    """Compare the converted dataset with its source; returns problems found."""
    problems = []
    info = read_json(out / INFO_PATH)
    if info.get("codebase_version") != TARGET_VERSION:
        problems.append(f"info.json codebase_version is {info.get('codebase_version')!r}")
    if info.get("total_episodes") != len(src.episodes):
        problems.append("info.json total_episodes changed")

    episodes = pq.read_table(out / EPISODES_PATH.format(chunk_index=0, file_index=0)).to_pylist()
    if [e["episode_index"] for e in episodes] != list(range(len(src.episodes))):
        problems.append("meta/episodes is not one row per episode in order")
        return problems

    tables: dict[tuple[int, int], pa.Table] = {}
    for episode, source_path in zip(episodes, src.data_files):
        i = episode["episode_index"]
        key = (episode["data/chunk_index"], episode["data/file_index"])
        if key not in tables:
            tables[key] = pq.read_table(out / DATA_PATH.format(chunk_index=key[0], file_index=key[1]))
        table = tables[key]
        rows = table.filter(pc.equal(table["episode_index"], i))
        source = pq.read_table(source_path)
        if rows.num_rows != source.num_rows:
            problems.append(f"episode {i}: {rows.num_rows} rows in its data file, source has "
                            f"{source.num_rows}")
            continue
        if not rows.equals(source):
            problems.append(f"episode {i}: rows differ from the source")
        first_index = rows["index"][0].as_py() if rows.num_rows else None
        if first_index != episode["dataset_from_index"]:
            problems.append(f"episode {i}: dataset_from_index {episode['dataset_from_index']} but "
                            f"first row index is {first_index}")

        for key in src.video_keys:
            span = episode[f"videos/{key}/to_timestamp"] - episode[f"videos/{key}/from_timestamp"]
            expected = episode["length"] / src.fps
            if abs(span - expected) > VIDEO_SPAN_TOLERANCE_FRAMES / src.fps:
                problems.append(f"episode {i} {key}: video span {span:.3f}s, expected "
                                f"{expected:.3f}s for {episode['length']} frames")

    # Every frame of every source video must be in the concatenated files.
    for key in src.video_keys:
        files = sorted({(e[f"videos/{key}/chunk_index"], e[f"videos/{key}/file_index"])
                        for e in episodes})
        converted = sum(_frame_count(out / VIDEO_PATH.format(video_key=key, chunk_index=c,
                                                             file_index=f)) for c, f in files)
        source = sum(_frame_count(p) for p in src.video_files[key])
        if converted != source:
            problems.append(f"{key}: {converted} frames after conversion, {source} before")
    return problems


def _frame_count(path: Path) -> int:
    """Count video packets without decoding (one packet per frame)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return sum(1 for packet in container.demux(stream) if packet.size)


# ---------------------------------------------------------------------------

def convert(input_dir: Path, output_dir: Path, data_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
            video_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB, check: bool = True) -> Path:
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise ConversionError(f"output {output_dir} already exists; choose another or remove it")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ConversionError("output must not be inside the input dataset")

    src = SourceDataset(input_dir)
    log.info(f"{input_dir}: {len(src.episodes)} episodes, {sum(e['length'] for e in src.episodes)} "
             f"frames, cameras {src.video_keys or 'none'}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        log.info("info.json / tasks")
        convert_info(src, work, data_mb, video_mb)
        convert_tasks(src, work)
        log.info("data files")
        data_meta = convert_data(src, work, data_mb)
        video_meta = {}
        for key in src.video_keys:
            log.info(f"videos: {key}")
            video_meta[key] = convert_videos_of_camera(src, work, key, video_mb)
        log.info("episode metadata and stats")
        convert_episodes(src, work, data_meta, video_meta)
        copy_extra_files(src, work)
        if check:
            log.info("verifying against the source")
            problems = verify(src, work)
            if problems:
                raise ConversionError(f"{len(problems)} verification problem(s): "
                                      + "; ".join(problems[:5]))
        work.rename(output_dir)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    log.info(f"wrote {output_dir}")
    return output_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a local LeRobot dataset from v2.1 to v3.0.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="v2.1 dataset root")
    parser.add_argument("--output", type=Path, default=None,
                        help="v3.0 dataset root to create (default: <input>_v30)")
    parser.add_argument("--data-file-size-in-mb", type=int, default=DEFAULT_DATA_FILE_SIZE_IN_MB,
                        help=f"target size of each data parquet (default {DEFAULT_DATA_FILE_SIZE_IN_MB})")
    parser.add_argument("--video-file-size-in-mb", type=int, default=DEFAULT_VIDEO_FILE_SIZE_IN_MB,
                        help=f"target size of each video file (default {DEFAULT_VIDEO_FILE_SIZE_IN_MB})")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip comparing the result with the source")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output = args.output or args.input.with_name(args.input.name + "_v30")
    try:
        convert(args.input, output, args.data_file_size_in_mb, args.video_file_size_in_mb,
                check=not args.no_verify)
    except ConversionError as exc:
        log.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
