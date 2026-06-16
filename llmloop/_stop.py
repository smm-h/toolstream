"""Stop orphaned opencode processes spawned by llmloop.

Usage:
    llmloop-stop          # list and terminate all opencode child processes
    llmloop-stop --all    # terminate ALL opencode processes (not just children)
"""

from __future__ import annotations

import os
import signal
import sys
import time


def _find_opencode_processes(*, all_users: bool = False) -> list[dict[str, str | int]]:
    """Find running opencode processes.

    When all_users is False, only finds processes whose parent is the current
    process tree (i.e., likely spawned by llmloop). When True, finds all
    opencode processes owned by the current user.
    """
    results = []
    my_uid = os.getuid()

    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat_path = f"/proc/{pid}/status"
            with open(stat_path) as f:
                status = f.read()

            uid_line = [line for line in status.splitlines() if line.startswith("Uid:")]
            if not uid_line:
                continue
            proc_uid = int(uid_line[0].split()[1])
            if proc_uid != my_uid:
                continue

            cmdline_path = f"/proc/{pid}/cmdline"
            with open(cmdline_path, "rb") as f:
                cmdline_raw = f.read()
            cmdline_parts = cmdline_raw.decode("utf-8", errors="replace").split("\0")
            cmdline = " ".join(p for p in cmdline_parts if p)

            if "opencode" not in cmdline:
                continue

            # Must be an "opencode run" invocation with --format json (spawned by llmloop)
            is_llmloop_child = "--format" in cmdline and "json" in cmdline and "run" in cmdline

            if not all_users and not is_llmloop_child:
                continue

            results.append({
                "pid": pid,
                "cmdline": cmdline,
            })
        except (OSError, ValueError, IndexError):
            continue

    return results


def stop(*, all_processes: bool = False) -> int:
    """Find and gracefully stop opencode processes.

    Returns the number of processes terminated.
    """
    procs = _find_opencode_processes(all_users=all_processes)

    if not procs:
        print("No opencode processes found.")
        return 0

    print(f"Found {len(procs)} opencode process(es):")
    for p in procs:
        print(f"  PID {p['pid']}: {p['cmdline'][:120]}")

    terminated = 0
    for p in procs:
        pid = p["pid"]
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"  Sent SIGTERM to PID {pid}")
            terminated += 1
        except ProcessLookupError:
            print(f"  PID {pid} already exited")
        except PermissionError:
            print(f"  Permission denied for PID {pid}")

    if terminated > 0:
        # Give processes a moment to exit gracefully
        time.sleep(2)

        # Check for stragglers and escalate to SIGKILL
        for p in procs:
            pid = p["pid"]
            try:
                os.kill(pid, 0)  # check if still alive
                os.kill(pid, signal.SIGKILL)
                print(f"  Sent SIGKILL to PID {pid} (did not exit after SIGTERM)")
            except ProcessLookupError:
                pass
            except PermissionError:
                pass

    print(f"Terminated {terminated} process(es).")
    return terminated


def main() -> None:
    all_flag = "--all" in sys.argv[1:]
    stop(all_processes=all_flag)


if __name__ == "__main__":
    main()
