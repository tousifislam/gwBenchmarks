"""Rebuild comparison artifacts + CHANGELOG from saved model dirs (all 22)."""
import json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import reporting as R
import run_all as RA

HERE = Path(__file__).resolve().parent
results = []
for md in sorted((HERE / "models").glob("NN_*")):
    sc = json.loads((md / "scorecard.json").read_text())
    npz = np.load(md / "saved_model" / "errors.npz")
    results.append(dict(scorecard=sc, err_tr=npz["err_train"], err_va=npz["err_val"],
                        tr_idx=npz["train_idx"], val_idx=npz["val_idx"]))
print(f"loaded {len(results)} approaches")
R.final_plots(results)
RA.write_changelog(results)
best = min(results, key=lambda x: x["scorecard"]["loss"])["scorecard"]
print("best:", best["approach"], round(best["loss"], 4))
print("categories:", sorted(set(r["scorecard"]["category"] for r in results)))
print("reparams:", sorted(set(r["scorecard"]["parameterization"] for r in results)))
