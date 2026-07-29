"""Stop one distributed AC calibration campaign on the current machine.

The launcher creates nested PowerShell/Python processes and the watchdog creates
``multiprocessing`` children. Killing only a launcher PID can therefore leave
workers alive. This command discovers the complete campaign process tree,
includes detached spawn children recorded by the queue, and verifies shutdown.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil


_SPAWN_PARENT_PATTERN = re.compile(r"\bparent_pid=(\d+)\b")
_CAMPAIGN_LAUNCHER_PATTERN = re.compile(
    r"launch_[^\s\"']*campaigns?\.py\s+launch\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    name: str
    command_line: str
    create_time: float


def _command_line(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(part) for part in value)
    return str(value or "")


def _snapshot_processes() -> dict[int, ProcessRecord]:
    records: dict[int, ProcessRecord] = {}
    for process in psutil.process_iter(
        ["pid", "ppid", "name", "cmdline", "create_time"]
    ):
        try:
            info = process.info
            record = ProcessRecord(
                pid=int(info["pid"]),
                ppid=int(info.get("ppid") or 0),
                name=str(info.get("name") or ""),
                command_line=_command_line(info.get("cmdline")),
                create_time=float(info.get("create_time") or 0.0),
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError, TypeError):
            continue
        records[record.pid] = record
    return records


def _own_process_tree() -> set[int]:
    excluded = {os.getpid()}
    try:
        parent = psutil.Process(os.getpid()).parent()
        while parent is not None:
            excluded.add(parent.pid)
            parent = parent.parent()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    return excluded


def _recorded_watchdog_pids(
    queue: Path | None,
    marker: str,
    hostname: str,
) -> set[int]:
    """Read watchdog PIDs written by this host, including dead parent PIDs."""

    if queue is None:
        return set()
    command_dir = queue / "_commands"
    try:
        paths = list(command_dir.glob(f"watchdog_{hostname}_*.txt"))
    except OSError as exc:
        print(f"WARNING: cannot inspect {command_dir}: {exc}", file=sys.stderr)
        return set()

    marker_lower = marker.lower()
    pids: set[int] = set()
    for path in paths:
        try:
            command = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if marker_lower not in command.lower():
            continue
        match = re.search(r"_(\d+)\.txt$", path.name)
        if match:
            pids.add(int(match.group(1)))
    return pids


def _expand_target_pids(
    records: dict[int, ProcessRecord],
    marker: str,
    recorded_watchdog_pids: set[int],
    excluded_pids: set[int],
) -> set[int]:
    """Find marker processes, their launchers, descendants, and spawn children."""

    marker_lower = marker.lower()
    targets = {
        record.pid
        for record in records.values()
        if record.pid not in excluded_pids
        and marker_lower in record.command_line.lower()
    }

    for pid in tuple(targets):
        parent_pid = records[pid].ppid
        while parent_pid in records and parent_pid not in excluded_pids:
            parent = records[parent_pid]
            if _CAMPAIGN_LAUNCHER_PATTERN.search(parent.command_line):
                targets.add(parent_pid)
            parent_pid = parent.ppid

    changed = True
    while changed:
        changed = False
        parent_candidates = targets | recorded_watchdog_pids
        for record in records.values():
            if record.pid in targets or record.pid in excluded_pids:
                continue
            is_descendant = record.ppid in targets
            spawn_parents = {
                int(value)
                for value in _SPAWN_PARENT_PATTERN.findall(record.command_line)
            }
            is_recorded_spawn = bool(spawn_parents & parent_candidates)
            if is_descendant or is_recorded_spawn:
                targets.add(record.pid)
                changed = True
    return targets


def _same_process(record: ProcessRecord) -> psutil.Process | None:
    try:
        process = psutil.Process(record.pid)
        if abs(process.create_time() - record.create_time) > 0.01:
            return None
        return process
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return None


def _kill_targets(
    records: dict[int, ProcessRecord],
    targets: set[int],
) -> None:
    roots = sorted(
        pid for pid in targets if records[pid].ppid not in targets
    )
    for pid in roots:
        completed = subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        output = (completed.stdout or completed.stderr).strip()
        if output:
            print(output)

    # A second PID-specific pass catches a child that detached during taskkill.
    for pid in sorted(targets, reverse=True):
        process = _same_process(records[pid])
        if process is None:
            continue
        try:
            process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue


def stop_campaign(args: argparse.Namespace) -> int:
    marker = args.marker.strip()
    if not marker:
        raise SystemExit("--marker cannot be empty")
    queue = Path(args.queue) if args.queue else None
    excluded = _own_process_tree()
    hostname = socket.gethostname()
    deadline = time.monotonic() + max(1.0, float(args.timeout))
    pass_number = 0

    while True:
        pass_number += 1
        records = _snapshot_processes()
        watchdog_pids = _recorded_watchdog_pids(
            queue,
            marker,
            hostname,
        )
        targets = _expand_target_pids(
            records,
            marker,
            watchdog_pids,
            excluded,
        )
        if not targets:
            print(
                f"STOP OK: no campaign processes remain on {hostname} "
                f"for marker {marker!r}."
            )
            return 0

        print(
            f"Pass {pass_number}: terminating {len(targets)} process(es): "
            + ", ".join(str(pid) for pid in sorted(targets))
        )
        if args.dry_run:
            return 0
        _kill_targets(records, targets)
        if time.monotonic() >= deadline:
            break
        time.sleep(1.0)

    records = _snapshot_processes()
    watchdog_pids = _recorded_watchdog_pids(queue, marker, hostname)
    remaining = _expand_target_pids(
        records,
        marker,
        watchdog_pids,
        excluded,
    )
    if not remaining:
        print(
            f"STOP OK: no campaign processes remain on {hostname} "
            f"for marker {marker!r}."
        )
        return 0

    print("STOP FAILED: these campaign processes remain:", file=sys.stderr)
    for pid in sorted(remaining):
        record = records[pid]
        print(
            f"  pid={record.pid} ppid={record.ppid} "
            f"name={record.name} cmd={record.command_line}",
            file=sys.stderr,
        )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", required=True)
    parser.add_argument(
        "--queue",
        default=None,
        help="Exact queue path; enables cleanup of detached workers.",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(stop_campaign(build_parser().parse_args(argv)))


if __name__ == "__main__":
    main()
