# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Where each episode came from, and the robot calibration it was recorded with.

The rosbag converters add these keys to every episode in meta/episodes.jsonl
(v3.0: columns of meta/episodes/*.parquet):

    source_recording    the recording's folder name, e.g.
                        20251103-3CO2_20260908-092948-054650_20260908-093149-245261
    source_mcap         [storage file name, ...] the episode's span lies in
    source_mcap_sha256  [sha256 of each of those files]
    robot_id            the folder name's robot prefix (20251103-3CO2), or None
    source_start_ns     start / end of the episode's segment in the recording's
    source_end_ns       log time, nanoseconds since the Unix epoch
    calibration         meta/calibration/<robot_id>_<sha256[:8]>.json, or None
    calibration_sha256  sha256 of that file

calibration.json is an attachment of each MCAP. Every MCAP of one robot
carries the same file until the robot is recalibrated, so it is stored once
per distinct content, byte for byte as attached. Nothing else attached to the
MCAP is read (it also carries the device's certificates).
"""

import hashlib
import json
import re
import struct
from pathlib import Path

CALIBRATION_ATTACHMENT = "calibration.json"
CALIBRATION_DIR = "meta/calibration"

# <robot>_<YYYYMMDD>...: 20251103-3CO2_20260908-092948-..., 4ACO_20261026113416
_ROBOT_PREFIX = re.compile(r"^([A-Za-z0-9-]+)_\d{8}")

_MCAP_MAGIC = b"\x89MCAP0\r\n"
_OP_ATTACHMENT, _OP_ATTACHMENT_INDEX, _OP_SUMMARY_OFFSET = 0x09, 0x0A, 0x0E
_FOOTER_SIZE = 1 + 8 + 8 + 8 + 4   # opcode, length, summary_start, summary_offset_start, crc


def robot_id_of(recording_name):
    m = _ROBOT_PREFIX.match(recording_name or "")
    return m.group(1) if m else None


def file_sha256(path, block=8 * 2**20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                return h.hexdigest()
            h.update(chunk)


def storage_files(bag_dir, fallback):
    """[(path, start_ns, end_ns)] of a rosbag2 directory's storage files, from
    metadata.yaml; times are None where it does not give them. A directory
    without usable metadata.yaml is just `fallback` (the converter opens that
    file directly then)."""
    import yaml
    try:
        meta = yaml.safe_load((Path(bag_dir) / "metadata.yaml").read_text(encoding="utf-8"))
        info = meta["rosbag2_bagfile_information"]
        names = list(info.get("relative_file_paths") or [])
    except (OSError, yaml.YAMLError, KeyError, TypeError):
        return [(Path(fallback), None, None)]
    if not names:
        return [(Path(fallback), None, None)]
    spans = {}
    for f in info.get("files") or []:
        try:
            start = int(f["starting_time"]["nanoseconds_since_epoch"])
            spans[f["path"]] = (start, start + int(f["duration"]["nanoseconds"]))
        except (KeyError, TypeError, ValueError):
            continue
    return [(Path(bag_dir) / n, *spans.get(n, (None, None))) for n in names]


def files_in_span(files, start_ns, end_ns):
    """The storage files a segment overlaps; all of them when their spans are unknown."""
    if any(f[1] is None for f in files):
        return list(files)
    return [f for f in files if f[1] <= end_ns and start_ns <= f[2]] or list(files)


def _read_record(fh):
    head = fh.read(9)
    if len(head) < 9:
        raise ValueError("truncated record")
    opcode, length = struct.unpack("<BQ", head)
    return opcode, fh.read(length)


def _string(buf, pos):
    (n,) = struct.unpack_from("<I", buf, pos)
    return buf[pos + 4:pos + 4 + n].decode("utf-8"), pos + 4 + n


def mcap_attachment(path, name):
    """The bytes of the attachment called `name`, or None if there is none.

    Read through the summary's attachment index, so only that record is read,
    not the file. An MCAP without a summary (not closed properly) is reported
    as having no attachment rather than scanned end to end."""
    with open(path, "rb") as fh:
        if fh.read(8) != _MCAP_MAGIC:
            raise ValueError(f"{path} is not an MCAP file")
        size = fh.seek(0, 2)
        if size < 8 + _FOOTER_SIZE + 8:
            return None
        fh.seek(size - 8 - _FOOTER_SIZE)
        footer = fh.read(_FOOTER_SIZE)
        if fh.read(8) != _MCAP_MAGIC:
            return None
        summary_start, summary_offset_start = struct.unpack_from("<QQ", footer, 9)
        if not summary_start:
            return None
        footer_at = size - 8 - _FOOTER_SIZE

        # The summary offsets say where the attachment indexes are; without
        # them, walk the summary section (small: indexes and statistics only).
        start, end = summary_start, (summary_offset_start or footer_at)
        if summary_offset_start:
            fh.seek(summary_offset_start)
            while fh.tell() < footer_at:
                opcode, body = _read_record(fh)
                if opcode == _OP_SUMMARY_OFFSET and body[0] == _OP_ATTACHMENT_INDEX:
                    group_start, group_length = struct.unpack_from("<QQ", body, 1)
                    start, end = group_start, group_start + group_length
                    break
            else:
                return None
        fh.seek(start)
        while fh.tell() < end:
            opcode, body = _read_record(fh)
            if opcode != _OP_ATTACHMENT_INDEX:
                continue
            offset = struct.unpack_from("<Q", body, 0)[0]
            found, _ = _string(body, 40)   # offset, length, log_time, create_time, data_size
            if found != name:
                continue
            fh.seek(offset)
            opcode, record = _read_record(fh)
            if opcode != _OP_ATTACHMENT:
                raise ValueError(f"{path}: attachment index points at record type {opcode}")
            pos = 16                                    # log_time, create_time
            _, pos = _string(record, pos)               # name
            _, pos = _string(record, pos)               # media_type
            (n,) = struct.unpack_from("<Q", record, pos)
            return record[pos + 8:pos + 8 + n]
    return None


class CalibrationStore:
    """Writes each distinct calibration.json once under <root>/meta/calibration."""

    def __init__(self, root):
        self.root = Path(root)
        self._by_sha = {}

    def add(self, data, robot_id):
        """Store `data`; returns (path relative to the dataset root, sha256)."""
        sha = hashlib.sha256(data).hexdigest()
        rel = self._by_sha.get(sha)
        if rel is None:
            stem = re.sub(r"[^A-Za-z0-9._-]", "_", robot_id) if robot_id else "unknown"
            rel = f"{CALIBRATION_DIR}/{stem}_{sha[:8]}.json"
            path = self.root / rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
            self._by_sha[sha] = rel
        return rel, sha


def prune_calibration(root, episodes):
    """Remove calibration files no episode refers to any more (after episodes
    were deleted); returns the names removed."""
    folder = Path(root) / CALIBRATION_DIR
    if not folder.is_dir():
        return []
    used = {Path(e["calibration"]).name for e in episodes if e.get("calibration")}
    removed = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name not in used:
            path.unlink()
            removed.append(path.name)
    return removed


def check_calibration_json(data):
    """Raise ValueError unless `data` is a JSON object (what a calibration file is)."""
    if not isinstance(json.loads(data), dict):
        raise ValueError("calibration.json is not a JSON object")
