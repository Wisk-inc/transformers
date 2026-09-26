# Copyright 2026 Nathan. Apache-2.0.
"""
The cell ends on purpose in several places, and only some of those are errors.

Launching the background run ends the cell -- the notebook has nothing left to do -- and it used to do
so with ``raise SystemExit``, which marimo shows as a red TRACEBACK. A run that had started perfectly
looked like a crash. These pin how each kind of stop looks in each place the cell runs: marimo,
Jupyter/IPython, and the plain Python process the background run is.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from naturev1 import follow, launch, status, stop
from naturev1.launch import _alive


HERE = Path(__file__).resolve().parents[1]


def _halt_source() -> str:
    """The cell's own ``_halt``, exactly as written there."""
    cell = (HERE / "colab_naturev1.py").read_text()
    for node in ast.parse(cell).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_halt":
            return "import sys\n\n" + ast.get_source_segment(cell, node) + "\n"
    raise AssertionError("the cell has no _halt")


@pytest.mark.parametrize("kind, code", [("success", 0), ("warn", 1), ("danger", 1)])
def test_a_plain_process_exits_with_the_status_the_supervisor_reads(tmp_path, kind, code):
    """0 only for success: the supervisor resumes a run that stopped part-way, and not a finished one."""
    script = tmp_path / "stop.py"
    script.write_text(_halt_source() + f"_halt('the message', kind={kind!r})\nprint('NOT REACHED')\n")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == code
    assert "the message" in done.stdout + done.stderr
    assert "NOT REACHED" not in done.stdout
    assert "Traceback" not in done.stderr


def test_jupyter_shows_the_message_without_a_traceback():
    IPython = pytest.importorskip("IPython")
    from IPython.utils.capture import capture_output

    shell = IPython.core.interactiveshell.InteractiveShell.instance()
    shell.run_cell(_halt_source())
    with capture_output() as shown:
        result = shell.run_cell("_halt('training in the background', kind='success')\nprint('NOT REACHED')")
    text = shown.stdout + shown.stderr
    assert "training in the background" in text
    assert "NOT REACHED" not in text, "the rest of the cell must not run"
    assert "Traceback" not in text and "SystemExit" not in text and "An exception has occurred" not in text
    assert result.error_in_exec is not None, "Run All must still stop at this cell"


@pytest.mark.parametrize("kind", ["success", "warn"])
def test_marimo_shows_a_note_and_not_a_traceback(tmp_path, kind):
    pytest.importorskip("marimo")
    body = textwrap.indent(_halt_source() + f"_halt('training in the background', kind={kind!r})\n"
                           "print('NOT REACHED')\n", "    ")
    notebook = tmp_path / "notebook.py"
    notebook.write_text(f"import marimo\n\napp = marimo.App()\n\n\n@app.cell\ndef _():\n{body}    return\n\n\n"
                        "if __name__ == '__main__':\n    app.run()\n")
    out = tmp_path / "out.ipynb"
    done = subprocess.run([sys.executable, "-m", "marimo", "export", "ipynb", str(notebook), "-o", str(out),
                           "--include-outputs"], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and "failed" not in (done.stdout + done.stderr), done.stdout + done.stderr
    outputs = [o for cell in json.loads(out.read_text())["cells"] for o in cell.get("outputs", [])]
    assert not [o for o in outputs if o["output_type"] == "error"], "marimo would show a TRACEBACK"
    shown = json.dumps(outputs)
    assert "training in the background" in shown and kind in shown, "the message, as a coloured note"
    assert "NOT REACHED" not in shown, "the rest of the cell must not run"


SLOW_START = textwrap.dedent('''
    import time
    time.sleep(3)                       # importing torch, in miniature
    print("HARDWARE", flush=True)
    print("a GPU", flush=True)
    while True:
        time.sleep(0.2)
''')


def test_follow_waits_for_a_new_run_and_not_for_a_running_one(tmp_path, capsys):
    fake = tmp_path / "fake_pipeline.py"
    fake.write_text(SLOW_START)
    home = tmp_path / "data"
    launch(directory=home, script=fake, supervise=False)
    try:
        follow(directory=home, wait=30)
        shown = capsys.readouterr().out
        assert "HARDWARE" in shown and "a GPU" in shown, "a fresh run's first lines are waited for"
        began = time.time()
        follow(directory=home, wait=30)
        shown = capsys.readouterr().out
        assert time.time() - began < 3, "a run that has already written something is shown at once"
        assert "a GPU" in shown
    finally:
        stop(directory=home, wait=20)


def test_a_run_just_launched_is_never_reported_stopped(tmp_path):
    """For a moment after it starts, a process shows an empty command line; that is not "stopped"."""
    fake = tmp_path / "fake_pipeline.py"
    fake.write_text("import time\ntime.sleep(5)\n")
    for trial in range(15):
        home = tmp_path / f"data{trial}"
        launch(directory=home, script=fake, stall_minutes=1.0, poll_seconds=0.5)
        try:
            for _ in range(20):
                assert status(home) is not None, f"launch {trial}: a starting run was reported as stopped"
                time.sleep(0.005)
        finally:
            stop(directory=home, wait=20)


def test_a_process_that_exited_is_not_running_even_before_it_is_reaped():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        assert _wait_for_zombie(process.pid), "the child should exit and wait, unreaped"
        assert not _alive(process.pid)
    finally:
        process.wait()


def _wait_for_zombie(pid, seconds=20.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
        if state == "Z":
            return True
        time.sleep(0.05)
    return False
