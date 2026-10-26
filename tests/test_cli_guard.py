"""Scripts that take no options must not run when handed one, and must not clobber single-run results.

`python watchdog/drip_test.py --help` used to execute a pre-registered test and rewrite its committed result,
because the script ignored argv entirely. For a test whose validity rests on having been run once, that is a
real hazard. `evaluation.cli.guard` closes it.
"""
import pytest

from evaluation.cli import RERUN, guard


def test_help_prints_the_docstring_and_runs_nothing(capsys):
    with pytest.raises(SystemExit) as e:
        guard("WHAT THIS SCRIPT DOES", ["--help"])
    assert e.value.code == 0
    assert "WHAT THIS SCRIPT DOES" in capsys.readouterr().out


def test_short_help_flag_also_works():
    with pytest.raises(SystemExit) as e:
        guard("doc", ["-h"])
    assert e.value.code == 0


def test_unknown_argument_is_refused_rather_than_ignored():
    with pytest.raises(SystemExit) as e:
        guard("doc", ["--bogus"])
    assert e.value.code != 0
    assert "bogus" in str(e.value.code)


def test_no_arguments_is_the_normal_path():
    guard("doc", [])  # must not raise


def test_existing_protected_output_blocks_the_run(tmp_path):
    out = tmp_path / "preregistered.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        guard("doc", [], protected=(out,))
    msg = str(e.value.code)
    assert "refusing to overwrite" in msg and out.name in msg


def test_rerun_overrides_the_block(tmp_path):
    out = tmp_path / "preregistered.json"
    out.write_text("{}", encoding="utf-8")
    guard("doc", [RERUN], protected=(out,))  # must not raise


def test_absent_protected_output_does_not_block(tmp_path):
    guard("doc", [], protected=(tmp_path / "not_created_yet.json",))


def test_allowed_extra_flags_pass_through():
    guard("doc", ["--verbose"], allow=("--verbose",))
    with pytest.raises(SystemExit):
        guard("doc", ["--verbose"])


@pytest.mark.parametrize("script", ["watchdog/drip_test.py", "watchdog/reversal_test.py",
                                    "watchdog/lateburst_test.py", "scripts/real_flow_rule.py"])
def test_single_run_scripts_are_wired_to_the_guard(script):
    """A new single-run script must not silently skip the protection."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / script).read_text(encoding="utf-8")
    assert "from evaluation.cli import guard" in src, f"{script} does not import the guard"
    assert "guard(__doc__" in src, f"{script} does not call the guard"
    assert "protected=" in src, f"{script} calls the guard without declaring its outputs"


@pytest.mark.parametrize("script", ["scripts/build_dashboard.py", "scripts/consolidate_results.py",
                                    "scripts/make_figures.py", "watchdog/posthoc_summary.py"])
def test_regenerable_scripts_refuse_unknown_arguments(script):
    """These rebuild derived artefacts, so re-running is fine and they take no `protected=`.

    They must still print usage for --help instead of executing: `--help` on posthoc_summary.py used to
    regenerate the summary it is meant to describe.
    """
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parent.parent / script).read_text(encoding="utf-8")
    assert "from evaluation.cli import guard" in src, f"{script} does not import the guard"
    assert "guard(__doc__" in src, f"{script} does not call the guard"


@pytest.mark.parametrize("script", ["scripts/build_dashboard.py", "scripts/make_figures.py",
                                    "watchdog/posthoc_summary.py", "watchdog/drip_test.py"])
def test_guard_is_callable_before_any_heavy_import(script):
    """The guard must be reachable without the project's optional dependencies being importable."""
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parent.parent / script).read_text(encoding="utf-8")
    gi = src.index("from evaluation.cli import guard")
    mi = src.index("def main()")
    assert gi < mi, f"{script} imports the guard after main() is defined"


def test_every_no_option_script_in_the_repo_is_guarded():
    """A sweep, so a new script in any directory cannot quietly reintroduce the hole.

    synthetic/calibrate.py was missed by the first pass because that pass only looked in scripts/, watchdog/
    and evaluation/. `--help` on it ran a full calibration.

    A script passes if it uses argparse, uses the shared guard, or handles --help itself (pnl_decompose.py
    takes positional arguments, so it parses them by hand and checks --help explicitly).
    """
    from pathlib import Path as _P
    root = _P(__file__).resolve().parent.parent
    offenders = []
    for d in ("scripts", "watchdog", "evaluation", "synthetic", "training", "env", "data"):
        for f in (root / d).glob("*.py"):
            if f.name == "__init__.py":
                continue
            src = f.read_text(encoding="utf-8")
            if "def main(" not in src:
                continue
            handles_help = '"--help" in' in src or "'--help' in" in src
            if "argparse" in src or "from evaluation.cli import guard" in src or handles_help:
                continue
            offenders.append(str(f.relative_to(root)))
    assert not offenders, (
        "these scripts take no options and have no guard, so any argument is ignored and they just run: "
        + ", ".join(offenders))
