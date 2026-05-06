from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import train_and_score_closed_form  # noqa: E402


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    models_dir = work_dir / "models"

    approaches = []

    # Physics-informed stitched inspiral/ringdown (smooth tanh transition)
    for qparam in ["q", "eta", "delta_m"]:
        for deg in [1, 2]:
            approaches.append(
                dict(
                    name=f"IMR stitch exp (x={qparam}, deg={deg})",
                    cat="physics-informed closed forms",
                    qparam=qparam,
                    amp_kind="exp_stitch",
                    transition_w=25.0,
                    poly_degree=deg,
                )
            )

    # Matched asymptotic / composite (sharper transition)
    for qparam in ["q", "eta", "delta_m"]:
        for deg in [1, 2]:
            approaches.append(
                dict(
                    name=f"Composite stitch sharp (x={qparam}, deg={deg})",
                    cat="matched asymptotic / composite",
                    qparam=qparam,
                    amp_kind="exp_stitch_sharp",
                    transition_w=10.0,
                    poly_degree=deg,
                )
            )

    # Functional form optimization (gaussian-modulated stitched form)
    for qparam in ["q", "eta", "delta_m"]:
        for deg in [1, 2]:
            approaches.append(
                dict(
                    name=f"Gauss-mod stitch (x={qparam}, deg={deg})",
                    cat="functional form optimization",
                    qparam=qparam,
                    amp_kind="gauss_mod",
                    transition_w=25.0,
                    poly_degree=deg,
                )
            )

    assert len(approaches) == 18

    for i, cfg in enumerate(approaches, start=1):
        model_dir = models_dir / (
            f"{i:02d}_"
            + cfg["name"]
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace(",", "")
            .replace("+", "_")
            .replace("-", "_")
            .replace("=", "_")
        )
        if (model_dir / "scorecard.json").exists():
            continue
        train_and_score_closed_form(
            work_dir=work_dir,
            model_dir=model_dir,
            approach_number=i,
            display_name=cfg["name"],
            category=cfg["cat"],
            qparam=cfg["qparam"],
            amp_kind=cfg["amp_kind"],
            poly_degree=int(cfg["poly_degree"]),
            transition_w=float(cfg["transition_w"]),
        )


if __name__ == "__main__":
    main()

