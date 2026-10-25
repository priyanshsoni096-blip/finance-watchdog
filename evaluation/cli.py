"""Argument guards for scripts that take no options, and overwrite protection for single-run results.

Several scripts here were written to take no arguments at all, so they ignored whatever was passed and ran.
`python watchdog/drip_test.py --help` therefore executed a pre-registered test and rewrote its committed
result instead of printing usage. For a test whose whole validity rests on having been run once, that is a
real hazard, not a cosmetic one.

`guard()` gives those scripts three properties:

  * `--help` / `-h` prints the module docstring and exits 0, running nothing.
  * an unrecognised argument exits 2 with a message, rather than being silently ignored.
  * a script that declares `protected` outputs refuses to overwrite them once they exist, unless `--rerun`
    is passed explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

RERUN = "--rerun"


def guard(doc: str | None, argv: list[str] | None = None, *, protected: tuple[Path, ...] = (),
          allow: tuple[str, ...] = ()) -> None:
    """Validate argv for a script that takes no options. Exits on --help, a bad argument, or a blocked rerun.

    `protected` are result files the script would overwrite. If any already exist, the script stops unless
    `--rerun` was passed, so a pre-registered result cannot be replaced by accident.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    name = Path(sys.argv[0]).name if sys.argv else "script"

    if "--help" in args or "-h" in args:
        print((doc or "").strip() or f"{name}: no help available")
        raise SystemExit(0)

    known = {RERUN, *allow}
    unknown = [a for a in args if a not in known]
    if unknown:
        allowed = ", ".join(sorted(known)) if known else "none"
        raise SystemExit(f"{name}: unrecognised argument(s): {' '.join(unknown)}\n"
                         f"this script takes no options (accepted: {allowed}); use --help for what it does")

    existing = [p for p in protected if p.exists()]
    if existing and RERUN not in args:
        listed = "\n  ".join(str(p) for p in existing)
        raise SystemExit(
            f"{name}: refusing to overwrite results that already exist:\n  {listed}\n"
            f"These record a run that was meant to happen once. Pass {RERUN} if you really intend to replace "
            f"them, or move them aside first.")
