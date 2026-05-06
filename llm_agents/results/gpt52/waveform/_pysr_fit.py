from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True, help="Path to X .npy")
    ap.add_argument("--y", required=True, help="Path to y .npy (1D)")
    ap.add_argument("--out", required=True, help="Output expressions .json")
    ap.add_argument("--maxsize", type=int, default=20)
    ap.add_argument("--niterations", type=int, default=80)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X = np.load(args.x).astype(np.float32)
    y = np.load(args.y).astype(np.float32)

    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=int(args.niterations),
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "log", "exp", "sin", "cos"],
        maxsize=int(args.maxsize),
        populations=10,
        procs=int(args.procs),
        random_state=int(args.seed),
        model_selection="best",
        loss="loss(prediction, target) = (prediction - target)^2",
    )
    model.fit(X, y)

    expressions: list[dict] = []
    eqs = model.equations_
    for _, row in eqs.iterrows():
        expressions.append(
            {
                "equation": str(row.get("equation", "")),
                "sympy_format": str(row.get("sympy_format", "")),
                "lambda_format": str(row.get("lambda_format", "")),
                "complexity": float(row.get("complexity", 0.0)),
                "loss": float(row.get("loss", 0.0)),
                "score": float(row.get("score", 0.0)) if "score" in row else None,
                "pick": str(row.get("pick", "")),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(expressions, indent=2) + "\n")


if __name__ == "__main__":
    main()

