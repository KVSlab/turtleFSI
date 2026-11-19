import numpy as np
import pytest
import json

from turtleFSI.cases.flag_campaign import case_plan, check_history


def test_grid25_matches_plotted_parameters():
    cases = case_plan(plan="grid25")
    assert len(cases) == len({c["name"] for c in cases}) == 25
    assert {(c["radius"], c["length"]) for c in cases} == {
        (r, l) for r in (.05, .05625, .0625, .06875, .075)
        for l in (.35, .3875, .425, .4625, .5)}
    assert all(c["dt"] == .01 and c["stride"] == 5 for c in cases)


def test_campaign_split_and_refinement():
    cases = case_plan()
    train = [c for c in cases if c["role"] == "training"]
    heldout = [c for c in cases if c["role"] == "heldout"]
    fine = [c for c in cases if c["role"] == "time_step_check"]
    assert (len(train), len(heldout), len(fine)) == (12, 2, 2)
    assert len({c["name"] for c in cases}) == len(cases)
    assert not {(c["radius"], c["length"]) for c in train} & {
        (c["radius"], c["length"]) for c in heldout}
    for c in fine:
        coarse = next(t for t in train if t["name"] == c["compare_to"])
        assert c["dt"] == coarse["dt"]/2
        assert c["stride"]*c["dt"] == coarse["stride"]*coarse["dt"]


def history():
    data = np.zeros(11, dtype=[(k, float) for k in
                    ("time", "tip_dx", "tip_dy", "drag", "lift", "min_J", "min_JK")])
    data["time"] = .01*np.arange(1, 12)
    data["min_J"] = data["min_JK"] = 1
    return data


def test_complete_history():
    check_history(history(), .1, .01)


@pytest.mark.parametrize("failure", ["truncated", "inverted", "nonfinite", "missing_step"])
def test_bad_history_rejected(failure):
    data = history()
    if failure == "truncated":
        data = data[:-1]
    elif failure == "inverted":
        data["min_J"][3] = -1
    elif failure == "nonfinite":
        data["lift"][3] = np.nan
    else:
        data["time"][3] = .05
    with pytest.raises((ValueError, AssertionError)):
        check_history(data, .1, .01)


def test_retry_skips_success_preserves_source_and_refines_only_failure(tmp_path, monkeypatch):
    from turtleFSI.cases.flag_campaign import prepare_retry
    from turtleFSI.cases.flag_geometry import DEFAULT_MAPPING
    monkeypatch.setattr("turtleFSI.cases.validate_flag_geometry.validate", lambda *args: {})
    source, target = tmp_path / "source", tmp_path / "retry"
    source.mkdir()
    cases = case_plan(plan="grid25")[:3]
    manifest = dict(cases=cases, mapping_version=DEFAULT_MAPPING, end_time=.1)
    original = json.dumps(manifest)
    (source / "campaign.json").write_text(original)
    (source / (cases[0]["name"]+".exitcode")).write_text("0\n")
    (source / (cases[1]["name"]+".exitcode")).write_text("1\n")
    folder = source / "results" / cases[0]["name"]
    folder.mkdir(parents=True)
    h = history()
    np.savetxt(folder / "reference_history.csv", np.column_stack([h[k] for k in h.dtype.names]),
               delimiter=",", header=",".join(h.dtype.names), comments="")
    result = prepare_retry(source, target, .005)
    assert [c["name"] for c in result["cases"]] == [c["name"] for c in cases[1:]]
    assert result["cases"][0]["dt"] == .005
    assert result["cases"][0]["stride"] == 10
    assert result["cases"][1]["dt"] == .01
    assert result["cases"][1]["stride"] == 5
    assert (source / "campaign.json").read_text() == original
    assert (source / (cases[1]["name"]+".exitcode")).read_text() == "1\n"
    assert result["restart_policy"] == "from_time_zero"


def test_shell_runner_continues_after_failed_batch(tmp_path):
    """Exercise real batch scheduling with fake resource wrapper and solvers."""
    import os
    import subprocess
    from pathlib import Path
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "systemd-run"
    wrapper.write_text('#!/bin/bash\nwhile [[ "$1" != limits ]]; do shift; done\nshift\nexec "$@"\n')
    wrapper.chmod(0o755)
    fake = bin_dir / "python"
    fake.write_text('''#!/bin/bash
if [[ "$1" == -m ]]; then
    if [[ "$3" == prepare ]]; then
        printf 'first\\t0.05\\t0.35\\t0.01\\t5\\nsecond\\t0.05\\t0.4\\t0.01\\t5\\nthird\\t0.05\\t0.5\\t0.01\\t5\\n' > "$4/cases.tsv"
        exit 0
    fi
    exit 99
fi
while [[ "$1" != --sub-folder ]]; do shift; done
shift
echo "attempted $1"
[[ "$1" != first ]]
''')
    fake.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "turtleFSI/cases/run_flag_training.sh"
    result = subprocess.run(["bash", str(script), str(tmp_path), "1", ".1", ".01", "1m", "5", "grid25"],
                            env=dict(os.environ, PATH=str(bin_dir)+":"+os.environ["PATH"]),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stderr
    output = next(tmp_path.glob("flag-training.*"))
    assert [(output / (n+".exitcode")).read_text().strip() for n in ("first", "second", "third")] == ["1", "0", "0"]
    assert "All queued cases attempted" in result.stdout
