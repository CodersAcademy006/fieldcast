"""Does TabPFN still beat the baselines with the weekend flag removed? Prints Spearman, best-day lift and how often the pick is a Saturday or Sunday."""
import json

import numpy as np
from scipy.stats import spearmanr

import model

obs, w = model.load()
tr, te = w[w.date < model.CUT], w[w.date >= model.CUT]
wk = te.date.dt.isocalendar().week.to_numpy()
out = {}
for flag in (True, False):
    feats = [f for f in model.FEATS if flag or f != "weekend"]
    for g in ("butterfly", "bird"):
        yt = te[f"{g}_species"].to_numpy()
        pred = np.expm1(model.reg(tr[feats], np.log1p(tr[f"{g}_species"])).predict(te[feats]))
        lift, wkend = [], []
        for k in np.unique(wk):
            i = np.where(wk == k)[0]
            if len(i) >= 5 and yt[i].mean() > 0:
                j = i[np.argmax(pred[i])]
                lift.append(yt[j] / yt[i].mean())
                wkend.append(te.iloc[j]["weekend"])
        out[f"{g}_{'with' if flag else 'without'}_weekend"] = {
            "spearman": round(float(spearmanr(pred, yt)[0]), 3), "best_day_lift": round(float(np.mean(lift)), 3),
            "pick_is_weekend": f"{int(sum(wkend))} of {len(wkend)} weeks"}
        print(g, flag, out[f"{g}_{'with' if flag else 'without'}_weekend"], flush=True)
json.dump(out, open("ablation.json", "w"), indent=1)
