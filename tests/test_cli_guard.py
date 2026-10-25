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
