#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Convert ROS2 bags to LeRobot v2.1, keeping only the segments the QC review passed.

This wraps convert_rosbag_to_lerobot.py. The base script splits every bag into
segments at the operator's X/Y button presses and converts each of them. This
script matches each segment to a row of the sign-desk review export
(`review_result.csv`) and records only the segments whose 段结论 is 有效; a
segment that is 无效, missing from the export, or not identified with
certainty is skipped.

How segments are matched
------------------------
By start time, not by position. Each review row carries the segment's
`start_log_time_ns`, which comes from the same recorder event stream as the X
press in the bag, and the two agree to well under a millisecond. A segment is
matched to the closest row of the same recording within one second.

Position would not survive the operator: a stray Y tap ends a segment early,
pressing X again splits one in two, and three Y taps throw a segment away.
Each of those shifts every later segment *index* by one, so from that point a
counted index would judge segments against the wrong review rows. Timestamps
are unaffected. A segment with no review row close enough is skipped, and the
report lists the rows that never matched so the gap is visible.

The CSV records no MCAP path, so the task name is inferred from the bag's
path. A bag is expected at either:

  <root>/<任务名称>/<recording>.mcap                       # folder names the task
  <root>/<date>-<device>_<start>_<end>/<recording>.mcap    # folder names the session

The recording's own name carries the start time (`rec_YYYYMMDD_HHMMSS_*.mcap`);
the session folder's timestamp is only a fallback, because folder clocks drift
by a second or so.

Only segments the operator completed count: one whose X was replaced by
another X never ended, so the sign desk does not have a row for it. Such a
segment is dropped here too, since the export cannot vouch for it.

Filtering happens while the bag is read: a rejected segment's frames are
discarded as they arrive and its video packets are not kept, so it is never
encoded. The bag is still read in full, because the later X/Y presses are only
known by scanning it.

Usage
-----
    conda activate rosbag2lerobot
    unset PYTHONPATH

    python scripts/convert_rosbag_to_lerobot_with_valid_check.py \
        --multibag \
        --input_directory ./data/bags \
        --review-path ./data/reviews \
        --output username/dataset_name \
        --task "Fold the black short-sleeve shirt"

The dataset is written to $HF_LEROBOT_HOME/<output>, as in the base script.

Options added on top of the base script
    --isCheckReview True|False   filter by the review export (default: True)
    --review-path PATH           CSV file, or a directory searched for *.csv;
                                 required when checking
    --review-report PATH         JSON report (default <dataset>/review_report.json)

Every other option is the base script's and is passed through unchanged,
e.g. --head_resolution 1920x960:

    python scripts/convert_rosbag_to_lerobot_with_valid_check.py --help
"""

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# The base script lives in this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_rosbag_to_lerobot as base  # noqa: E402

REQUIRED_COLUMNS = ('任务名称', 'index', '段结论')

# Why a segment was not recorded. Kept machine-readable for the report; the
# text is only for the log.
REVIEW_REJECTED = 'review_rejected'
REVIEW_MISSING_SEGMENT = 'review_missing_segment'
REVIEW_UNKNOWN_CONCLUSION = 'review_unknown_conclusion'
REVIEW_MISSING_TASK = 'review_missing_task'
REVIEW_TASK_UNMATCHED = 'review_task_unmatched'
REVIEW_TASK_AMBIGUOUS = 'review_task_ambiguous'
# Segment outcomes: saved, skipped (a review reason above), or discarded for
# one of the base script's own reasons (restarted, too short, failed taps,
# open_at_bag_end).

REASON_TEXT = {
    REVIEW_REJECTED: 'the review export marks this segment 无效',
    REVIEW_MISSING_SEGMENT: 'the review export has no row starting at this time',
    REVIEW_UNKNOWN_CONCLUSION: '段结论 is neither 有效 nor 无效',
    REVIEW_MISSING_TASK: 'the review export has no rows for this recording',
    REVIEW_TASK_UNMATCHED: 'the recording path does not name a task',
    REVIEW_TASK_AMBIGUOUS: 'the recording path matches more than one task',
}

CONCLUSION_REASON = {'有效': None, '无效': REVIEW_REJECTED}

# A segment's X press is matched to the review row whose start_log_time_ns is
# closest, within this window. The two come from the same event stream and
# agree to well under a millisecond in practice; a second is a generous
# allowance for device clock offset. Wide enough to survive small drift, far
# narrower than the shortest segment, so it cannot pick a neighbour.
SEGMENT_MATCH_TOLERANCE_NS = 1_000_000_000


def _nanoseconds(value):
    """Parse start_log_time_ns / end_log_time_ns.

    The export writes these as a spreadsheet-quoted literal, ="1792985805677254834",
    so the quotes and equals sign have to be stripped.
    """
    if value is None:
        return None
    text = value.strip().strip('="')
    return int(text) if text.isdigit() else None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ('true', '1', 'yes'):
        return True
    if normalized in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('expected True or False')


class ReviewTable:
    """The review export: one row per (任务名称, index), with its time range.

    The whole export is validated while loading: a malformed row, or two files
    disagreeing about one segment, stops the run before conversion instead of
    leaving some segments judged from partial data.
    """

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        if self.path.is_file():
            files = [self.path]
        elif self.path.is_dir():
            files = sorted(p for p in self.path.rglob('*.csv')
                           if p.is_file() and not p.name.startswith('._'))
        else:
            raise ValueError(f'--review-path is not a file or directory: {self.path}')
        if not files:
            raise ValueError(f'no CSV files found under {self.path}')

        self.records = {}
        self.tasks = set()
        self.files = [str(p) for p in files]
        for file in files:
            self._load(file)
        if not self.records:
            raise ValueError(f'no segments found in {self.path}')

        self.rejected = {(r['task_name'], r['index']) for r in self.records.values()
                         if r['reason'] == REVIEW_REJECTED}
        self.rejected_indices = defaultdict(set)
        for task, index in self.rejected:
            self.rejected_indices[task].add(index)

        # Per task, once every file has been read.
        self.by_task = defaultdict(list)
        for record in self.records.values():
            self.by_task[record['task_name']].append(record)
        for records in self.by_task.values():
            records.sort(key=lambda r: r['index'])

    def _load(self, file):
        try:
            text = file.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:
            text = file.read_text(encoding='gb18030')
        reader = csv.DictReader(io.StringIO(text))
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f'{file}: missing columns {missing}')
        for row in reader:
            if not any(value for value in row.values()):
                continue
            task = (row['任务名称'] or '').strip()
            raw_index = (row['index'] or '').strip()
            if not task or not re.fullmatch(r'\d+', raw_index):
                raise ValueError(f'{file}:{reader.line_num}: invalid 任务名称 or index')
            conclusion = (row['段结论'] or '').strip()
            record = {
                'task_name': task,
                'index': int(raw_index),
                'conclusion': conclusion,
                'reason': CONCLUSION_REASON.get(conclusion, REVIEW_UNKNOWN_CONCLUSION),
                'failure_reason': (row.get('不通过原因') or '').strip(),
                'start_ns': _nanoseconds(row.get('start_log_time_ns')),
                'end_ns': _nanoseconds(row.get('end_log_time_ns')),
                'source': f'{file}:{reader.line_num}',
            }
            key = (task, record['index'])
            previous = self.records.get(key)
            if previous and previous['conclusion'] != conclusion:
                raise ValueError(
                    f'{task} index={record["index"]} is {previous["conclusion"]!r} in '
                    f'{previous["source"]} but {conclusion!r} in {record["source"]}'
                )
            self.records.setdefault(key, record)
            self.tasks.add(task)

    def lookup(self, task, start_time):
        """The review row for a segment that started at `start_time` (bag seconds).

        Matching on the recording clock rather than on the running index is
        what makes the export robust to the operator: a stray Y tap ends a
        segment early, an X restart splits one in two, and either way the
        segment *indexes* from then on no longer line up with the CSV. The
        timestamps do.
        """
        if start_time is None:
            return self._missing(task, None, 'no recording time')
        start_ns = int(round(start_time * 1e9))
        best, best_gap = None, None
        for record in self.by_task.get(task, ()):
            if record['start_ns'] is None:
                continue
            gap = abs(record['start_ns'] - start_ns)
            if best_gap is None or gap < best_gap:
                best, best_gap = record, gap
        if best is not None and best_gap <= SEGMENT_MATCH_TOLERANCE_NS:
            return dict(best)
        return self._missing(task, best, best_gap)

    @staticmethod
    def _missing(task, nearest, gap):
        return {'task_name': task, 'index': nearest['index'] if nearest else None,
                'conclusion': None, 'reason': REVIEW_MISSING_SEGMENT,
                'failure_reason': None, 'source': None,
                'nearest_gap_s': None if gap is None else round(gap / 1e9, 3)}


def task_candidates(bag_file):
    """Task names the recording may belong to.

    The recording's own timestamp is authoritative and comes first; the
    session folder's is only a fallback.
    """
    path = Path(bag_file)
    folder, parent = path.parent.name, path.parent.parent.name
    recording = re.fullmatch(r'rec_(\d{8})_(\d{6})(?:_\d+)?\.(?:mcap|db3)', path.name)
    # Session folder written by the recorder:
    #   20260728-4ACO_20261026-113416-465493_20261026-115341-064512
    #   <date>-<device>_<start>_<end>
    # Group 1 is the device (4ACO). The sign desk names the task
    # <device>_<start timestamp>, i.e. 4ACO_20261026113416.
    session = re.fullmatch(
        r'\d{8}-(.+?)_(\d{8})-(\d{6})(?:-\d+)?(?:_\d{8}-\d{6}(?:-\d+)?)?', folder)
    candidates = []

    def add(name, rule):
        if name and not any(c['task_name'] == name for c in candidates):
            candidates.append({'task_name': name, 'rule': rule})

    if recording:
        device = session[1] if session else parent
        add(f'{device}_{recording[1]}{recording[2]}', 'recording_name')
    if session:
        add(f'{session[1]}_{session[2]}{session[3]}', 'session_folder')
    # The folder itself may already be the task name (a downloaded bag is
    # stored as <任务名称>/); candidates are tried in order, so this comes
    # after the more specific rules.
    add(folder, 'folder_name')
    named = re.fullmatch(r'(.+)_(\d{14})', folder)
    if named:
        add(named[1], 'folder_without_timestamp')
    if parent:
        add(f'{parent}_{folder}', 'parent_and_folder')
    return candidates


def resolve_task(bag_file, known_tasks):
    """(task_name or None, candidates, reason)."""
    candidates = task_candidates(bag_file)
    matches = [c for c in candidates if c['task_name'] in known_tasks]
    if len(matches) == 1:
        return matches[0]['task_name'], candidates, None
    if len(matches) > 1:
        return None, candidates, REVIEW_TASK_AMBIGUOUS
    # Nothing matched. If the path produced a concrete name, the export simply
    # has no rows for this recording; otherwise it did not identify a task well
    # enough to say that.
    if candidates and candidates[0]['rule'] in ('recording_name', 'folder_name'):
        return None, candidates, REVIEW_MISSING_TASK
    return None, candidates, REVIEW_TASK_UNMATCHED


class _DiscardingPacketBuffer(base.AllIdrVideoBuffer):
    """Packet buffer for a skipped segment: keeps nothing, so a rejected
    segment costs no memory for its video while the bag is read."""

    def add_packet(self, *args, **kwargs):
        pass


class ReviewedConverter(base.MultiVideoRosBag2LeRobotConverter):
    """The base converter, with segments filtered by the review export.

    The base script is not modified. Two of its methods are wrapped:

      _start_episode_capture  an X press opens a segment; its start time is
                              looked up in the review export
      _submit_episode         a stopped segment is handed over for saving;
                              a rejected one is cleared here instead

    and convert_single_bag marks bag boundaries.

    Matching is by the segment's start time, not by a running index (see
    ReviewTable.lookup): a stray Y tap cuts a segment short and an X restart
    splits one in two, so segment indexes stop lining up with the CSV, while
    the event timestamps do not.
    """

    def __init__(self, input_directory, output_repo_id, fps=45, crf=23,
                 reviews=None, report_path=None, **base_options):
        super().__init__(input_directory, output_repo_id, fps=fps, crf=crf, **base_options)
        self.reviews = reviews
        self.report_path = Path(report_path) if report_path else None
        self.report_bags = []
        self._bag = None
        self._segment = None

    # ---- per bag -------------------------------------------------------
    def convert_single_bag(self, rosbag, task_description, enforce_all, whole_bag=False):
        if whole_bag and self.reviews is not None:
            self.logger.warning(
                '--whole_bag has no X/Y segments, so the review export cannot be '
                'applied; each bag is converted as one episode'
            )
        self._begin_bag(rosbag)
        try:
            return super().convert_single_bag(rosbag, task_description, enforce_all, whole_bag)
        finally:
            if self._segment is not None:
                self._close(self._segment, 'discarded', 'open_at_bag_end')
                self._segment = None

    def _begin_bag(self, rosbag):
        bag_file = rosbag['bag_file']
        if self.reviews is None:
            task, candidates, reason = None, [], None
        else:
            task, candidates, reason = resolve_task(bag_file, self.reviews.tasks)
        self._bag = {
            'bag_name': rosbag['name'],
            'bag_file': str(Path(bag_file).resolve()),
            'review_task_name': task,
            'task_match': {'rule': candidates[0]['rule'] if candidates else None,
                           'candidates': candidates, 'reason': reason},
            'segments': [],
        }
        self.report_bags.append(self._bag)
        self._segment = None
        if self.reviews is not None:
            if task is None:
                self.logger.warning(
                    f'{bag_file}: no task found in the review export ({reason}); '
                    f'every segment of this bag will be skipped'
                )
            else:
                self.logger.info(f'{bag_file}: review task {task}')

    # ---- segments ------------------------------------------------------
    def _start_episode_capture(self, recent_packets, start_time):
        # An X press while recording means the base script throws the previous
        # unfinished segment away; it never reaches _submit_episode.
        if self._segment is not None:
            self._close(self._segment, 'discarded', 'restarted')

        review, reason = None, None
        if self.reviews is not None:
            task = self._bag['review_task_name']
            if task is None:
                reason = self._bag['task_match']['reason']
            else:
                review = self.reviews.lookup(task, start_time)
                reason = review['reason']
        self._segment = {
            'index': review['index'] if review else None,
            'start_time_s': start_time,
            'end_time_s': None,
            'status': 'recording',
            'review': review,
            'reason': reason,
        }
        if reason is None:
            return super()._start_episode_capture(recent_packets, start_time)

        self.logger.info(f'  segment at {start_time:.3f}s: skipped — {REASON_TEXT[reason]}'
                         + (f' (nearest review row is {review["nearest_gap_s"]}s away)'
                            if review and review.get('nearest_gap_s') is not None else ''))
        # The recording loop still adds frames to an episode buffer, so give it
        # an empty one; video packets are not kept at all.
        self.dataset.episode_buffer = self.dataset.create_episode_buffer()
        self._episode_ticks = []
        return _DiscardingPacketBuffer(
            root_dir=self.dataset.root, fps=self.dataset.fps, crf=self.crf
        )

    def _submit_episode(self, packet_buffer, start_time, end_time, task_description):
        segment, self._segment = self._segment, None
        if segment is None:
            # --whole_bag, or the review filter is off for this run.
            return super()._submit_episode(packet_buffer, start_time, end_time, task_description)

        segment['end_time_s'] = end_time
        if segment['reason'] is not None:
            self.dataset.clear_episode_buffer()
            packet_buffer.clear()
            self._episode_ticks = []
            self._close(segment, 'skipped')
            return None

        pending_before = len(self._pending)
        super()._submit_episode(packet_buffer, start_time, end_time, task_description)
        # The base script drops segments shorter than MIN_EPISODE_LENGTH.
        saved = len(self._pending) > pending_before
        self._close(segment, 'saved' if saved else 'discarded',
                    None if saved else 'too_short')
        return None

    def _close(self, segment, status, discard_reason=None):
        segment['status'] = status
        if discard_reason:
            segment['discard_reason'] = discard_reason
        self._bag['segments'].append(segment)

    # ---- report --------------------------------------------------------
    def write_report(self):
        """Write the report, or return None when there is nowhere to put it.

        The base script returns before creating the dataset when it finds no
        bags; with no --review-report path there is then no dataset root to
        write into, and the base script has already logged why.
        """
        if self.report_path is None and self.dataset is None:
            return None
        path = self.report_path or (Path(self.dataset.root) / 'review_report.json')
        checks = []
        for bag in self.report_bags:
            task = bag['review_task_name']
            if self.reviews is None or task not in self.reviews.tasks:
                continue
            task_rows = self.reviews.by_task.get(task, [])
            rejected = self.reviews.rejected_indices.get(task, set())
            matched = {s['index'] for s in bag['segments'] if s['index'] is not None}
            recorded = {s['index'] for s in bag['segments'] if s['status'] == 'saved'}
            checks.append({
                'bag_name': bag['bag_name'],
                'task_name': task,
                'review_rows': len(task_rows),
                'segments_in_bag': len(bag['segments']),
                'review_rows_matched': len(matched),
                # A row that never matched a segment means the CSV and the
                # recording disagree about this bag: worth looking at.
                'review_rows_never_matched': sorted(
                    r['index'] for r in task_rows if r['index'] not in matched),
                'rejected_in_export': len(rejected),
                'rejected_encountered': len(rejected & matched),
                'rejected_not_reached': sorted(rejected - matched),
                'rejected_recorded_by_mistake': sorted(rejected & recorded),
            })

        for bag in self.report_bags:
            counts = defaultdict(int)
            for segment in bag['segments']:
                counts[segment['status']] += 1
            bag['counts'] = dict(counts)

        report = {
            'schema_version': 1,
            'dataset_root': str(self.dataset.root) if self.dataset else None,
            'review_enabled': self.reviews is not None,
            'review_files': self.reviews.files if self.reviews else [],
            'episode_count': int(self.dataset.num_episodes) if self.dataset else 0,
            'checks': checks,
            'bags': self.report_bags,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                       encoding='utf-8')
        os.replace(tmp, path)
        return path

    def convert_all(self, *args, **kwargs):
        try:
            return super().convert_all(*args, **kwargs)
        finally:
            report_path = self.write_report()
            if report_path is not None:
                self.logger.info(f'Review report: {report_path}')


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Convert ROS2 bags to LeRobot v2.1, keeping only the segments '
                    'the QC review passed.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--isCheckReview', type=parse_bool, nargs='?', const=True,
                        default=True,
                        help='filter segments by the review export (default: True); '
                             '--isCheckReview False converts every segment')
    parser.add_argument('--review-path', default=None,
                        help='review CSV file, or a directory searched for *.csv; '
                             'required when checking')
    parser.add_argument('--review-report', type=Path, default=None,
                        help='JSON report path (default: <dataset>/review_report.json)')
    ours, passthrough = parser.parse_known_args(argv)
    if ours.isCheckReview and not ours.review_path:
        parser.error('--review-path is required when --isCheckReview is True')
    return ours, passthrough


def main(argv=None):
    ours, passthrough = parse_args(sys.argv[1:] if argv is None else list(argv))

    reviews = None
    if ours.isCheckReview:
        try:
            reviews = ReviewTable(ours.review_path)
        except ValueError as exc:
            sys.exit(f'error: {exc}')
        print(f'Review export: {len(reviews.records)} segments in {len(reviews.files)} '
              f'file(s), {len(reviews.tasks)} task(s)')

    return _run(passthrough, reviews, ours.review_report)


def _run(argv, reviews, report_path):
    """Run the base script's main() with ReviewedConverter in place of the
    converter class. The base script creates its converter inside main(), so
    the class is swapped there rather than re-implementing its argument
    parsing and logging setup here."""
    original = base.MultiVideoRosBag2LeRobotConverter
    base.MultiVideoRosBag2LeRobotConverter = ReviewedConverter
    original_init = ReviewedConverter.__init__

    def converter_init(self, input_directory, output_repo_id, fps=45, crf=23, **base_options):
        # base_options: whatever else the base script passes (e.g. output_sizes)
        original_init(self, input_directory, output_repo_id, fps=fps, crf=crf,
                      reviews=reviews, report_path=report_path, **base_options)
    ReviewedConverter.__init__ = converter_init
    # The base script parses sys.argv itself, and our options are not its
    # options, so only the arguments it understands are shown to it.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], *argv]
    try:
        base.main()
    finally:
        sys.argv = saved_argv
        ReviewedConverter.__init__ = original_init
        base.MultiVideoRosBag2LeRobotConverter = original


if __name__ == '__main__':
    main()
