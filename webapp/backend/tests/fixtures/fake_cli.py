"""Stand-in for `supply_chain_simulator.cli run`, used only by test_runs.py.

Mimics just enough of the real CLI's stdout contract (an `Experiment ID:`
line, one `Replication N/Total: delta=... winner=...` line per replication,
an `Output written to:` line, and matching exit codes) for
`app.services.run_launcher` to be tested without spawning the real CLI —
which would make real, billable OpenAI calls and take tens of seconds per
replication. Never imported directly; `run_launcher`'s injectable
`command_builder` invokes it as a subprocess, exactly like the real CLI.
"""

import sys
import time
from pathlib import Path

import yaml  # type: ignore[import-untyped]


def main() -> None:
    mode, config_path_arg = sys.argv[1], sys.argv[2]
    config_path = Path(config_path_arg)
    experiment = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    print(f"Experiment ID: {experiment['experiment_id']}", flush=True)

    if mode == "hang":
        time.sleep(30)
        return

    total = experiment["replications"]

    if mode == "fail":
        print(f"Replication 1/{total}: delta=-1.000000 winner=LLM", flush=True)
        print("ERROR: simulated failure", flush=True)
        sys.exit(3)

    output_root = (config_path.parent / experiment["output_root"]).resolve()
    result_dir = output_root / f"{experiment['experiment_id']}__fake_timestamp"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "manifest.json").write_text(
        f'{{"experiment_id": "{experiment["experiment_id"]}"}}', encoding="utf-8"
    )

    for i in range(1, total + 1):
        print(f"Replication {i}/{total}: delta=-1.000000 winner=LLM", flush=True)

    print(f"Output written to: {result_dir}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
