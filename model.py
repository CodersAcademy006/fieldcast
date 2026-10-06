"""Daily butterfly/bird sighting forecaster: TabPFN on weather + calendar, backtested on the last 12 months.
Run:  .venv/bin/python model.py backtest   -> prints metrics, writes metrics.json
      .venv/bin/python model.py forecast   -> writes forecast.json (next 7 days, from Open-Meteo forecast)"""
import json
import sys

import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier, TabPFNRegressor

LAT, LNG = 18.52, 73.86
CUT = pd.Timestamp("2025-10-05")  # train before, test on the last 12 months
WCOLS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "relative_humidity_2m_mean",
         "wind_speed_10m_max", "shortwave_radiation_sum"]
FEATS = ["doy_sin", "doy_cos", "weekend", "year", "rain7", "rain30", "tmax7", "tmin7"] + WCOLS
REG_CKPT = ("Prior-Labs/TabPFN-v2-reg", "tabpfn-v2-regressor-09gpqh39.ckpt")
CLS_CKPT = ("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier.ckpt")  # open v2 weights, no gated terms


def features(w):
    w = w.sort_values("date").reset_index(drop=True)
    d = w["date"]
    w["doy_sin"], w["doy_cos"] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25), np.cos(2 * np.pi * d.dt.dayofyear / 365.25)
    w["weekend"] = (d.dt.dayofweek >= 5).astype(int)
    w["year"] = d.dt.year + d.dt.dayofyear / 365.25
    w["rain7"] = w["precipitation_sum"].rolling(7, min_periods=1).sum()
    w["rain30"] = w["precipitation_sum"].rolling(30, min_periods=1).sum()
    w["tmax7"] = w["temperature_2m_max"].rolling(7, min_periods=1).mean()
    w["tmin7"] = w["temperature_2m_min"].rolling(7, min_periods=1).mean()
    return w


def load():
    obs, w = pd.read_parquet("data/obs.parquet"), pd.read_parquet("data/weather.parquet")
    w = features(w)
    for g in ("butterfly", "bird"):
        s = obs[obs.group == g].groupby("date")["sci"].nunique()
        w[f"{g}_species"] = w["date"].map(s).fillna(0)
    return obs, w


def reg(x, y, n=None):
    m = TabPFNRegressor(model_path=hf_hub_download(*REG_CKPT), device="cpu", n_estimators=n or 4)
    return m.fit(x, y)


def cls(x, y):
    return TabPFNClassifier(model_path=hf_hub_download(*CLS_CKPT), device="cpu").fit(x, y)


def top_species(obs, g, k):
    t = obs[obs.group == g].groupby("sci").size().sort_values(ascending=False).head(k).index
    common = obs.dropna(subset=["common"]).drop_duplicates("sci").set_index("sci")["common"]
    return [(s, common.get(s, s)) for s in t]


def backtest():
    obs, w = load()
    tr, te = w[w.date < CUT], w[w.date >= CUT]
    out = {"train_days": len(tr), "test_days": len(te), "activity": {}, "species_auc": {}}
    for g in ("butterfly", "bird"):
        y = np.log1p(tr[f"{g}_species"])
        yt = te[f"{g}_species"].to_numpy()
        clim = tr.groupby(tr.date.dt.isocalendar().week)[f"{g}_species"].mean()
        p = {"climatology": te.date.dt.isocalendar().week.map(clim).fillna(tr[f"{g}_species"].mean()).to_numpy(),
             "gradient_boosting": np.expm1(HistGradientBoostingRegressor(random_state=0).fit(tr[FEATS], y).predict(te[FEATS])),
             "tabpfn": np.expm1(reg(tr[FEATS], y).predict(te[FEATS]))}
        wk = te.date.dt.isocalendar().week.to_numpy()
        res = {}
        for name, pred in p.items():
            lift = []  # pick the predicted-best day of each week: how does it compare to the week's average day?
            for k in np.unique(wk):
                i = np.where(wk == k)[0]
                if len(i) >= 5 and yt[i].mean() > 0:
                    lift.append(yt[i][np.argmax(pred[i])] / yt[i].mean())
            res[name] = {"spearman": round(float(spearmanr(pred, yt)[0]), 3),
                         "mae": round(float(np.mean(np.abs(pred - yt))), 3),
                         "best_day_lift": round(float(np.mean(lift)), 3)}
        out["activity"][g] = res
    for g, k in (("butterfly", 8), ("bird", 6)):
        for sci, common in top_species(obs, g, k):
            days = set(obs[obs.sci == sci].date)
            yb = w["date"].isin(days).astype(int)
            ytr, yte = yb[w.date < CUT], yb[w.date >= CUT]
            if yte.nunique() < 2:
                continue
            clim = tr.assign(y=ytr).groupby(tr.date.dt.isocalendar().week)["y"].mean()
            base = te.date.dt.isocalendar().week.map(clim).fillna(ytr.mean())
            auc_t = roc_auc_score(yte, cls(tr[FEATS], ytr).predict_proba(te[FEATS])[:, 1])
            out["species_auc"][common] = {"tabpfn": round(float(auc_t), 3), "climatology": round(float(roc_auc_score(yte, base)), 3),
                                          "days_logged_train": int(ytr.sum())}
            print(common, out["species_auc"][common], flush=True)
    json.dump(out, open("metrics.json", "w"), indent=1)
    print(json.dumps(out["activity"], indent=1))


def forecast():
    obs, w = load()
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params={
        "latitude": LAT, "longitude": LNG, "timezone": "Asia/Kolkata", "past_days": 30, "forecast_days": 7,
        "daily": ",".join(WCOLS)})
    r.raise_for_status()
    f = pd.DataFrame(r.json()["daily"]).rename(columns={"time": "date"})
    f["date"] = pd.to_datetime(f["date"])
    last = w.date.max()
    f = features(pd.concat([w[w.date > last - pd.Timedelta(days=30)][["date"] + WCOLS], f[f.date > last][["date"] + WCOLS]]))
    nxt = f[f.date > pd.Timestamp.today().normalize() - pd.Timedelta(days=1)].head(7).copy()
    out = {"generated": str(pd.Timestamp.today().date()), "days": []}
    for g in ("butterfly", "bird"):
        nxt[g] = np.expm1(reg(w[FEATS], np.log1p(w[f"{g}_species"])).predict(nxt[FEATS]))
    sp = {}
    for g, k in (("butterfly", 5), ("bird", 4)):
        for sci, common in top_species(obs, g, k):
            yb = w["date"].isin(set(obs[obs.sci == sci].date)).astype(int)
            sp[common] = cls(w[FEATS], yb).predict_proba(nxt[FEATS])[:, 1]
    for i, (_, r) in enumerate(nxt.iterrows()):
        out["days"].append({"date": str(r.date.date()), "weekday": r.date.day_name(), "tmax": round(float(r.temperature_2m_max), 1),
                            "rain_mm": round(float(r.precipitation_sum), 1), "butterfly_species": round(float(r.butterfly), 1),
                            "bird_species": round(float(r.bird), 1),
                            "likely": {k: round(float(v[i]), 2) for k, v in sorted(sp.items(), key=lambda kv: -kv[1][i])[:4]}})
    json.dump(out, open("forecast.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"backtest": backtest, "forecast": forecast}[sys.argv[1]]()
