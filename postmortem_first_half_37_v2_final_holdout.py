#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FIRST HALF 37 — V2 FINAL HOLDOUT POST-MORTEM
Diagnostic uniquement: aucun retrain, aucun tuning.

Analyse:
- intégrité FREEZE_LOCK / EVALUATION_LOCK / SHA256;
- performance globale + bootstrap 95%;
- drift des 8 features (coverage, SMD, KS, PSI);
- contribution univariée orientée par le coefficient;
- neutralisation feature par feature SANS refit;
- performance par jour, état du score, compétition;
- calibration par bandes;
- diagnostic automatique: panne générale vs feature breakdown vs sous-régimes.
"""
from __future__ import annotations

import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
FROZEN_DIR = BASE_DIR / "outputs" / "first_half_37" / "v2_frozen"
MODEL_PATH = FROZEN_DIR / "FIRST_HALF37_V2_FROZEN.joblib"
FREEZE_LOCK_PATH = FROZEN_DIR / "FIRST_HALF37_V2_FREEZE_LOCK.json"
DEV_PATH = BASE_DIR / "outputs" / "first_half_37" / "v2_robust_selection" / "First_Half_37_V2_DEVELOPMENT.csv"
HOLDOUT_DIR = BASE_DIR / "outputs" / "first_half_37" / "v2_final_holdout"
HOLDOUT_PATH = HOLDOUT_DIR / "First_Half_37_V2_FINAL_HOLDOUT.csv"
EVAL_DIR = HOLDOUT_DIR / "evaluation"
EVAL_LOCK_PATH = EVAL_DIR / "FIRST_HALF37_V2_HOLDOUT_EVALUATION_LOCK.json"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "first_half_37" / "v2_postmortem"

TARGET = "goal_after_37_before_ht"
DATE_COL = "kickoff_utc"
ID_COL = "match_id"
EPS = 1e-10
BOOT_N = 2000
BOOT_SEED = 87
FROZEN_BASE_RATE = 610 / 2105
EXPECTED_FEATURES = [
    "home_over25_freq", "shots_35_37", "away_shots_35_37", "away_xg_37",
    "subs_35_37", "home_corners_30_37", "total_xg_37", "score_37_away",
]


def now_utc_iso(): return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists(): raise FileNotFoundError(path)
    x = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x, dict): raise ValueError(f"JSON invalide: {path}")
    return x

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1-EPS)
    return np.log(p / (1-p))

def safe_roc(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))

def safe_pr(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    return float("nan") if len(np.unique(y)) < 2 else float(average_precision_score(y, p))

def ece(y, p, bins=10):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins+1); out = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p <= edges[i+1]) if i == bins-1 else (p < edges[i+1]))
        if m.any(): out += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(out)

def baseline_metrics(y, p0):
    q = np.full(len(y), p0, dtype=float)
    return float(brier_score_loss(y, q)), float(log_loss(y, np.clip(q, EPS, 1-EPS), labels=[0, 1]))

def metric_row(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    base = float(y.mean()); b0, ll0 = baseline_metrics(y, FROZEN_BASE_RATE)
    b = float(brier_score_loss(y, p)); ll = float(log_loss(y, np.clip(p, EPS, 1-EPS), labels=[0, 1]))
    return {
        "n": int(len(y)), "positives": int(y.sum()), "positive_rate": base,
        "mean_probability": float(p.mean()), "gap_meanp_minus_actual": float(p.mean()-base),
        "roc_auc": safe_roc(y,p), "pr_auc": safe_pr(y,p),
        "pr_lift_vs_prevalence": safe_pr(y,p)-base,
        "brier": b, "baseline_brier": b0, "brier_improvement": b0-b,
        "log_loss": ll, "baseline_logloss": ll0, "logloss_improvement": ll0-ll,
        "ece_10": ece(y,p,10),
    }

def verify_integrity():
    for p in (MODEL_PATH, FREEZE_LOCK_PATH, EVAL_LOCK_PATH, DEV_PATH, HOLDOUT_PATH):
        if not p.exists(): raise FileNotFoundError(p)
    freeze = load_json(FREEZE_LOCK_PATH); ev = load_json(EVAL_LOCK_PATH)
    if ev.get("status") != "FINAL_HOLDOUT_CONSUMED": raise RuntimeError("Holdout non marqué consommé.")
    model_sha = sha256_file(MODEL_PATH)
    if freeze.get("model_sha256") != model_sha or ev.get("model_sha256") != model_sha:
        raise RuntimeError("SHA modèle incohérent avec le gel/évaluation.")
    hold_sha = sha256_file(HOLDOUT_PATH)
    if ev.get("holdout_sha256") != hold_sha: raise RuntimeError("Holdout modifié depuis évaluation.")
    dev_sha = sha256_file(DEV_PATH)
    if freeze.get("development_sha256") and freeze["development_sha256"] != dev_sha:
        raise RuntimeError("DEVELOPMENT modifié depuis le gel.")
    bundle = joblib.load(MODEL_PATH)
    if list(bundle.get("features", [])) != EXPECTED_FEATURES: raise RuntimeError("TOP_8 du bundle inattendu.")
    return freeze, ev, dict(bundle)

def load_data():
    dev = pd.read_csv(DEV_PATH, low_memory=False); hold = pd.read_csv(HOLDOUT_PATH, low_memory=False)
    for name, df in (("DEV",dev),("HOLDOUT",hold)):
        req = set(EXPECTED_FEATURES+[TARGET,DATE_COL,ID_COL]); miss = sorted(req-set(df.columns))
        if miss: raise RuntimeError(f"{name}: colonnes absentes {miss}")
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce", format="mixed")
        if df[DATE_COL].isna().any(): raise RuntimeError(f"{name}: dates invalides")
        y = pd.to_numeric(df[TARGET], errors="coerce")
        if y.isna().any(): raise RuntimeError(f"{name}: target non résolu")
        df[TARGET] = y.astype(int)
    overlap = set(dev[ID_COL].astype(str)) & set(hold[ID_COL].astype(str))
    if overlap: raise RuntimeError(f"Leakage DEV/HOLDOUT: {len(overlap)} matchs")
    return dev.sort_values([DATE_COL,ID_COL]).reset_index(drop=True), hold.sort_values([DATE_COL,ID_COL]).reset_index(drop=True)

def transformed(bundle, df):
    pipe = bundle["base_model"]; X = df[bundle["features"]].apply(pd.to_numeric, errors="coerce")
    Zi = pipe.named_steps["imputer"].transform(X)
    Z = pipe.named_steps["scaler"].transform(Zi)
    m = pipe.named_steps["model"]
    return np.asarray(Z,float), np.asarray(m.coef_[0],float), float(m.intercept_[0])

def ks_stat(a,b):
    a=np.sort(np.asarray(a,float)); b=np.sort(np.asarray(b,float)); a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if not len(a) or not len(b): return float("nan")
    v=np.sort(np.unique(np.concatenate([a,b])))
    return float(np.max(np.abs(np.searchsorted(a,v,side="right")/len(a)-np.searchsorted(b,v,side="right")/len(b))))

def psi(dev, hold, bins=10):
    d=np.asarray(dev,float); h=np.asarray(hold,float); d=d[np.isfinite(d)]; h=h[np.isfinite(h)]
    if len(d)<10 or len(h)<10: return float("nan")
    e=np.unique(np.quantile(d,np.linspace(0,1,bins+1)))
    if len(e)<3: return 0.0
    e[0],e[-1]=-np.inf,np.inf
    dc,_=np.histogram(d,bins=e); hc,_=np.histogram(h,bins=e)
    dp=np.clip(dc/max(dc.sum(),1),1e-6,None); hp=np.clip(hc/max(hc.sum(),1),1e-6,None)
    return float(np.sum((hp-dp)*np.log(hp/dp)))

def feature_drift(dev, hold):
    rows=[]
    for f in EXPECTED_FEATURES:
        d=pd.to_numeric(dev[f],errors="coerce"); h=pd.to_numeric(hold[f],errors="coerce")
        dm=float(d.mean()) if d.notna().any() else np.nan; hm=float(h.mean()) if h.notna().any() else np.nan
        ds=float(d.std(ddof=0)) if d.notna().sum()>1 else np.nan
        smd=(hm-dm)/ds if pd.notna(ds) and ds>1e-12 else np.nan
        ps=psi(d.to_numpy(),h.to_numpy())
        level="HIGH" if ((pd.notna(ps) and ps>=.25) or (pd.notna(smd) and abs(smd)>=.5)) else ("MODERATE" if ((pd.notna(ps) and ps>=.10) or (pd.notna(smd) and abs(smd)>=.25)) else "LOW")
        rows.append({"feature":f,"dev_coverage":float(d.notna().mean()),"holdout_coverage":float(h.notna().mean()),"coverage_delta":float(h.notna().mean()-d.notna().mean()),"dev_mean":dm,"holdout_mean":hm,"standardized_mean_shift":smd,"ks_statistic":ks_stat(d,h),"psi":ps,"drift_level":level})
    order={"HIGH":0,"MODERATE":1,"LOW":2}; out=pd.DataFrame(rows); out["_o"]=out["drift_level"].map(order)
    return out.sort_values(["_o","psi"],ascending=[True,False]).drop(columns="_o").reset_index(drop=True)

def contribution_audit(bundle, dev, hold, alpha):
    Zd,coef,_=transformed(bundle,dev); Zh,coef2,intercept=transformed(bundle,hold)
    yd=dev[TARGET].to_numpy(int); yh=hold[TARGET].to_numpy(int)
    base_logit=intercept+Zh@coef; basep=sigmoid(base_logit+alpha); bm=metric_row(yh,basep)
    rows=[]
    for j,f in enumerate(EXPECTED_FEATURES):
        cd=Zd[:,j]*coef[j]; ch=Zh[:,j]*coef[j]
        da=safe_roc(yd,cd); ha=safe_roc(yh,ch)
        npred=sigmoid((base_logit-ch)+alpha); nm=metric_row(yh,npred)
        rg=nm["roc_auc"]-bm["roc_auc"]; bg=bm["brier"]-nm["brier"]; lg=bm["log_loss"]-nm["log_loss"]
        direction=int(np.isfinite(da) and np.isfinite(ha) and da>=.53 and ha<=.47)
        suspect=int((np.isfinite(ha) and ha<.47) and (rg>=.005 or bg>=.001 or lg>=.002))
        rows.append({"feature":f,"coefficient":float(coef[j]),"dev_contribution_auc":da,"holdout_contribution_auc":ha,"auc_delta_holdout_minus_dev":ha-da,"direction_break_flag":direction,"neutralized_roc_auc":nm["roc_auc"],"neutralization_roc_gain":rg,"neutralization_pr_gain":nm["pr_auc"]-bm["pr_auc"],"neutralization_brier_gain":bg,"neutralization_logloss_gain":lg,"strong_suspect":suspect})
    return pd.DataFrame(rows).sort_values(["strong_suspect","direction_break_flag","neutralization_roc_gain"],ascending=[False,False,False]).reset_index(drop=True)

def subgroup(df,p,col):
    t=df.copy(); t["_p"]=p; rows=[]
    for g,x in t.groupby(col,dropna=False):
        r=metric_row(x[TARGET].to_numpy(int),x["_p"].to_numpy(float)); r[col]=g; rows.append(r)
    return pd.DataFrame(rows)

def score_state(df):
    if "score_37_home" not in df.columns or "score_37_away" not in df.columns: return None
    h=pd.to_numeric(df["score_37_home"],errors="coerce"); a=pd.to_numeric(df["score_37_away"],errors="coerce"); out=[]
    for hh,aa in zip(h,a):
        if pd.isna(hh) or pd.isna(aa): out.append("missing")
        elif hh==0 and aa==0: out.append("0-0")
        elif hh==aa: out.append("draw_with_goals")
        elif hh>aa: out.append("home_lead")
        else: out.append("away_lead")
    return pd.Series(out,index=df.index)

def competition_col(df):
    for c in ["competition_name","league_name","tournament_name","competition","league","competition_id"]:
        if c in df.columns: return c
    return None

def prob_bands(df,p):
    edges=[0,.20,.24,.27,.30,.33,.36,.40,1.000001]; labels=["<20","20-24","24-27","27-30","30-33","33-36","36-40","40+"]
    t=pd.DataFrame({"y":df[TARGET].to_numpy(int),"p":p}); t["band"]=pd.cut(t["p"],bins=edges,labels=labels,include_lowest=True,right=False).astype(str)
    rows=[]
    for lab in labels:
        g=t[t["band"]==lab]
        if len(g): rows.append({"probability_band":lab,"n":len(g),"positives":int(g.y.sum()),"observed_rate":float(g.y.mean()),"mean_probability":float(g.p.mean()),"observed_minus_predicted":float(g.y.mean()-g.p.mean())})
    return pd.DataFrame(rows)

def bootstrap(y,p):
    rng=np.random.default_rng(BOOT_SEED); y=np.asarray(y,int); p=np.asarray(p,float); vals={k:[] for k in ["roc_auc","pr_lift","brier_improvement","logloss_improvement"]}
    for _ in range(BOOT_N):
        idx=rng.integers(0,len(y),len(y)); yy=y[idx]; pp=p[idx]
        if len(np.unique(yy))<2: continue
        prev=float(yy.mean()); b0,ll0=baseline_metrics(yy,FROZEN_BASE_RATE)
        vals["roc_auc"].append(safe_roc(yy,pp)); vals["pr_lift"].append(safe_pr(yy,pp)-prev); vals["brier_improvement"].append(b0-brier_score_loss(yy,pp)); vals["logloss_improvement"].append(ll0-log_loss(yy,np.clip(pp,EPS,1-EPS), labels=[0, 1]))
    out={}
    for k,a in vals.items():
        a=np.asarray(a,float); out[k]={"p2_5":float(np.percentile(a,2.5)),"median":float(np.percentile(a,50)),"p97_5":float(np.percentile(a,97.5)),"probability_above_zero":float((a>0).mean())}
    return out

def diagnosis(overall,daily,states,comp,contrib):
    de=daily[daily.n>=20]; se=states[states.n>=30] if len(states) else states; ce=comp[comp.n>=20] if len(comp) else comp
    dfrac=float((de.roc_auc>.5).mean()) if len(de) else np.nan; sfrac=float((se.roc_auc>.5).mean()) if len(se) else np.nan; cfrac=float((ce.roc_auc>.5).mean()) if len(ce) else np.nan
    suspects=contrib.loc[contrib.strong_suspect==1,"feature"].tolist(); breaks=contrib.loc[contrib.direction_break_flag==1,"feature"].tolist()
    broad=overall["roc_auc"]<=.52 and (np.isnan(dfrac) or dfrac<=.55) and (np.isnan(sfrac) or sfrac<=.60)
    diag="BROAD_SIGNAL_WEAKNESS" if broad else ("FEATURE_DIRECTION_BREAKDOWN" if len(suspects)>=2 or len(breaks)>=2 else "MIXED_SUBREGIME_FAILURE")
    return {"diagnosis":diag,"daily_eligible_n":int(len(de)),"daily_roc_above_050_fraction":None if np.isnan(dfrac) else dfrac,"score_state_eligible_n":int(len(se)),"score_state_roc_above_050_fraction":None if np.isnan(sfrac) else sfrac,"competition_eligible_n":int(len(ce)),"competition_roc_above_050_fraction":None if np.isnan(cfrac) else cfrac,"strong_suspect_features":suspects,"direction_break_features":breaks}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",default=str(DEFAULT_OUTPUT)); args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    print("="*120); print("FIRST HALF 37 — V2 FINAL HOLDOUT POST-MORTEM"); print("AUCUN RETRAIN / AUCUN TUNING"); print("="*120)
    freeze,ev,bundle=verify_integrity(); dev,hold=load_data(); alpha=float(bundle["calibration"]["alpha"])
    Z,coef,intercept=transformed(bundle,hold); rawp=sigmoid(intercept+Z@coef); p=sigmoid(logit(rawp)+alpha); y=hold[TARGET].to_numpy(int)
    overall=metric_row(y,p); boot=bootstrap(y,p); drift=feature_drift(dev,hold); contrib=contribution_audit(bundle,dev,hold,alpha)
    ddf=hold.copy(); ddf["date"]=ddf[DATE_COL].dt.strftime("%Y-%m-%d"); daily=subgroup(ddf,p,"date").sort_values("date")
    ss=score_state(hold)
    if ss is not None:
        sdf=hold.copy(); sdf["score_state_37"]=ss; states=subgroup(sdf,p,"score_state_37").sort_values("n",ascending=False)
    else: states=pd.DataFrame()
    cc=competition_col(hold); comp=subgroup(hold,p,cc).sort_values(["n","roc_auc"],ascending=[False,False]) if cc else pd.DataFrame()
    bands=prob_bands(hold,p); diag=diagnosis(overall,daily,states,comp,contrib)

    (out/"first_half_37_v2_postmortem_overall.json").write_text(json.dumps(overall,indent=2),encoding="utf-8")
    (out/"first_half_37_v2_postmortem_bootstrap.json").write_text(json.dumps(boot,indent=2),encoding="utf-8")
    drift.to_csv(out/"first_half_37_v2_postmortem_feature_drift.csv",index=False)
    contrib.to_csv(out/"first_half_37_v2_postmortem_feature_contributions.csv",index=False)
    daily.to_csv(out/"first_half_37_v2_postmortem_daily.csv",index=False)
    states.to_csv(out/"first_half_37_v2_postmortem_score_states.csv",index=False)
    comp.to_csv(out/"first_half_37_v2_postmortem_competitions.csv",index=False)
    bands.to_csv(out/"first_half_37_v2_postmortem_probability_bands.csv",index=False)

    report=[]
    report += ["# First Half 37 — V2 FINAL HOLDOUT POST-MORTEM","","## Discipline","","- Aucun retrain / tuning.","- Holdout final déjà consommé.","- Neutralisation feature = diagnostic sans refit.","- Ces 594 matchs ne peuvent plus être un nouveau holdout pristine.",""]
    report += ["## Overall","",pd.DataFrame([overall]).to_string(index=False),"","## Diagnostic automatique","",json.dumps(diag,indent=2,ensure_ascii=False),""]
    report += ["## Bootstrap 95%","",json.dumps(boot,indent=2,ensure_ascii=False),""]
    report += ["## Feature drift","",drift.to_string(index=False),"","## Feature contribution / neutralization","",contrib.to_string(index=False),""]
    report += ["## Daily","",daily.to_string(index=False),"","## Score states","",states.to_string(index=False) if len(states) else "UNAVAILABLE",""]
    c20=comp[comp.n>=20] if len(comp) else pd.DataFrame(); report += ["## Competitions n>=20","",c20.to_string(index=False) if len(c20) else "NONE / UNAVAILABLE",""]
    report += ["## Probability bands","",bands.to_string(index=False),"","## Suite","","Toute future V3 première mi-temps développée à partir de ce diagnostic devra être validée sur un autre bloc futur entièrement neuf."]
    (out/"first_half_37_v2_postmortem_report.md").write_text("\n".join(report),encoding="utf-8")
    manifest={"schema_version":"first_half_37_v2_postmortem_v1","created_at_utc":now_utc_iso(),"mode":"diagnostic_only_no_retrain_no_tuning","final_holdout_status":"CONSUMED","original_verdict":ev.get("verdict"),"development_rows":len(dev),"holdout_rows":len(hold),"model_sha256":sha256_file(MODEL_PATH),"holdout_sha256":sha256_file(HOLDOUT_PATH),"diagnosis":diag}
    (out/"first_half_37_v2_postmortem_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")

    print("POST-MORTEM V2 TERMINE")
    print(f"Diagnostic général : {diag['diagnosis']}")
    print(f"ROC holdout        : {overall['roc_auc']:.4f}")
    print(f"PR lift            : {overall['pr_lift_vs_prevalence']:+.4f}")
    print(f"Brier improvement  : {overall['brier_improvement']:+.4f}")
    print(f"LogLoss improvement: {overall['logloss_improvement']:+.4f}")
    print("Suspects forts      : " + (", ".join(diag["strong_suspect_features"]) or "aucun"))
    print("Direction breaks    : " + (", ".join(diag["direction_break_features"]) or "aucun"))
    print(f"Rapport             : {out/'first_half_37_v2_postmortem_report.md'}")
    print("IMPORTANT: aucun modèle n'a été réentraîné ou modifié.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
