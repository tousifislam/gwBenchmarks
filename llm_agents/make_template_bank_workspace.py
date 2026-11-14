#!/usr/bin/env python3
"""Build a self-contained agent workspace for the Template Bank Bench.

The workspace holds only public material, so an agent working inside it cannot
see the hidden targets, the evaluator, reference scores or other agents' runs:

    <workspace>/
    |-- AGENT_PROMPT.md          task description (llm_agents/agent_prompts/template_bank.md)
    |-- AGENTS.md, README.md     rules and quick start; name the Python interpreter to use
    |-- template_bank_core.py    copy of gwbenchmarks/benchmarks/template_bank_core.py
    |-- create_submission.py     starter submission (writes output/)
    |-- detection_statistic.py   starter statistic, max_i |z_i|^2
    |-- requirements.txt
    |-- data/                    f_amp.npy, Aref_weights.npy, bank_wf_params.npy (public only)
    `-- output/                  the agent's submission goes here

Usage (from the gwBenchmarks root):

    # Recommended: an isolated folder OUTSIDE the repository
    python llm_agents/make_template_bank_workspace.py /scratch/runs/my_run

    # In-repo layout used by the suite-wide agent loop (generate_prompt.py calls this)
    python llm_agents/make_template_bank_workspace.py llm_agents/results/<agent>/template_bank

Then start the agent inside the workspace, for example

    cd /scratch/runs/my_run && claude -p "Read AGENT_PROMPT.md, AGENTS.md, and README.md completely, \
then execute the benchmark autonomously. Work until every completion criterion is satisfied, all \
required files are written under output/, and TEMPLATE_BANK_BENCH_COMPLETE is printed."

and score it with ``python llm_agents/evaluate_template_bank.py --submission-dir /scratch/runs/my_run/output``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STARTERS = REPO_ROOT / "llm_agents" / "starters" / "template_bank"
CORE = REPO_ROOT / "gwbenchmarks" / "benchmarks" / "template_bank_core.py"
PROMPT = REPO_ROOT / "llm_agents" / "agent_prompts" / "template_bank.md"
DATA_DIR = REPO_ROOT / "datasets" / "template_bank"
# Only these files are public. bank_wf_params_test.npy (hidden targets) must never be copied.
PUBLIC_DATA = ("f_amp.npy", "Aref_weights.npy", "bank_wf_params.npy")


def build_workspace(target: Path, python: str, agent_label: str = "Template Bank",
                    agent: str = "agent", force: bool = False) -> Path:
    """Create the workspace at ``target`` and return its path."""

    target = target.resolve()
    if target.exists() and any(target.iterdir()):
        if not force:
            raise FileExistsError(f"{target} is not empty; pass --force to replace it.")
        shutil.rmtree(target)
    (target / "data").mkdir(parents=True)
    (target / "output").mkdir()

    for name in PUBLIC_DATA:
        source = DATA_DIR / name
        if not source.exists():
            raise FileNotFoundError(f"{source} is missing; download the dataset first (datasets/README.md).")
        shutil.copyfile(source, target / "data" / name)
    shutil.copyfile(CORE, target / "template_bank_core.py")
    for name in ("create_submission.py", "detection_statistic.py", "requirements.txt"):
        shutil.copyfile(STARTERS / name, target / name)
    for name in ("AGENTS.md", "README.md"):
        text = (STARTERS / name).read_text(encoding="utf-8")
        (target / name).write_text(text.replace("{PYTHON}", python), encoding="utf-8")

    # A neutral label by default, so the agent does not learn its model or condition from the prompt.
    prompt = PROMPT.read_text(encoding="utf-8").replace("{AGENT_LABEL}", agent_label).replace("{AGENT}", agent)
    (target / "AGENT_PROMPT.md").write_text(prompt, encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workspace", help="directory to create")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter the agent must use (needs numpy, scipy, lalsuite; numba optional)")
    parser.add_argument("--agent-label", default="Template Bank")
    parser.add_argument("--force", action="store_true", help="replace an existing non-empty workspace")
    args = parser.parse_args()
    path = build_workspace(Path(args.workspace), args.python, args.agent_label, force=args.force)
    print(f"Created Template Bank workspace: {path}")


if __name__ == "__main__":
    main()
