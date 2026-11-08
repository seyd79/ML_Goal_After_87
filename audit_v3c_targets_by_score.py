#!/usr/bin/env python3

# -*- coding: utf-8 -*-

from pathlib import Path

import json, math, sys

import joblib

import numpy as np

import pandas as pd

ROOT = Path(__file__).resolve().parent

if ROOT.name == 'data' and ROOT.parent.name == 'mnt':

       # quand le fichier est copié à la racine du projet, ROOT devient automatiquement le projet

       pass

LOW, HIGH = 0.55, 0.65

EXPECTED = {

       'OOS_684': (684, 157, 49),

       'BATCH1_279': (279, 55, 20),

       'BATCH2_221': (221, 58, 23),

       'RESERVE168_FINAL': (168, 39, 14),

}

def csv_read(path):

       tries = [

           dict(sep=',', encoding='utf-8-sig', engine='c'),

           dict(sep='\t', encoding='utf-8-sig', engine='c'),

           dict(sep=None, encoding='utf-8-sig', engine='python'),

           dict(sep=',', encoding='latin-1', engine='python'),

           dict(sep='\t', encoding='latin-1', engine='python'),

       ]

       last = None

       for kw in tries:

           try:

               d = pd.read_csv(path, **kw)

               if d.shape[1] > 1:

                   return d

           except Exception as e:

               last = e

       raise RuntimeError(f'Impossible de lire {path}: {last}')

def find_one(patterns):

       for pat in patterns:

           xs = list((ROOT/'outputs').rglob(pat))

           if xs:

               return sorted(xs, key=lambda p: len(str(p)))[0]

       raise FileNotFoundError(patterns)

def extract_38(obj):

       if isinstance(obj, list) and len(obj) == 38 and all(isinstance(x,str) for x in obj):

           return obj

       if isinstance(obj, dict):

           for v in obj.values():

               r = extract_38(v)

               if r: return r

       if isinstance(obj, list):

           for v in obj:

               r = extract_38(v)

               if r: return r

       return None

def target_series(df):

       for c in ['goal_after_87','target','y','label']:

           if c in df.columns:

               s = df[c]

               if pd.api.types.is_numeric_dtype(s):

                   return pd.to_numeric(s, errors='coerce').astype(float)

               m={'1':1,'0':0,'true':1,'false':0,'positive':1,'negative':0,'pos':1,'neg':0}

               return s.astype(str).str.strip().str.lower().map(m).astype(float)

       raise RuntimeError('Target introuvable')

def pcol_oos(df):

       for c in ['v3c_probability','p_v3c','v3c_prob','prob_v3c','probability_v3c','v3c_pred_proba']:

           if c in df.columns: return c

       for c in df.columns:

           cl=c.lower()

           if 'v3c' in cl and any(x in cl for x in ['prob','pred','score']):

               s=pd.to_numeric(df[c], errors='coerce')

               if s.notna().mean()>.8 and s.min()>=0 and s.max()<=1: return c

       raise RuntimeError('Probabilité V3-C OOS introuvable')

def wilson(k,n):

       if n==0: return (np.nan,np.nan)

       z=1.959963984540054; p=k/n; den=1+z*z/n

       cen=(p+z*z/(2*n))/den

       half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den

       return cen-half, cen+half

def add_score_fields(df):

       x=df.copy()

       h=pd.to_numeric(x['score_87_home'], errors='coerce')

       a=pd.to_numeric(x['score_87_away'], errors='coerce')

       d=h-a; t=h+a

       x['score_exact_87']=np.where(h.notna()&a.notna(),h.fillna(0).astype(int).astype(str)+'-'+a.fillna(0).astype(int).astype(str),'UNKNOWN')

       x['score_state_87']=np.select([d.eq(0),d.gt(0),d.lt(0)],['DRAW','HOME_LEADING','AWAY_LEADING'],default='UNKNOWN')

       x['margin_bucket_87']=np.select([d.abs().eq(0),d.abs().eq(1),d.abs().ge(2)],['DRAW','ONE_GOAL_MARGIN','TWO_PLUS_MARGIN'],default='UNKNOWN')

       x['total_goals_bucket_87']=np.select([t.eq(0),t.eq(1),t.eq(2),t.eq(3),t.ge(4)],['0','1','2','3','4+'],default='UNKNOWN')

       x['score_archetype_87']=np.select([h.eq(0)&a.eq(0),d.eq(0)&t.gt(0),d.abs().eq(1),d.abs().ge(2)],['0-0','DRAW_NONZERO','ONE_GOAL_MARGIN','TWO_PLUS_MARGIN'],default='UNKNOWN')

       return x

def summarize(sel, sample, group_col):

       base=sel['target'].mean()

       rows=[]

       for key,g in sel.groupby(group_col, dropna=False):

           n=len(g); k=int(g['target'].sum()); r=k/n; lo,hi=wilson(k,n)

           rows.append(dict(sample=sample, grouping=group_col, group=str(key), selected_n=n, selected_positives=k,

                            selected_rate=r, selected_base_rate=base, lift_vs_all_v3c_selected_pp=(r-base)*100,

                            ci95_low=lo, ci95_high=hi, fair_odds_empirical=(1/r if r>0 else np.nan), roi_at_3_20=r*3.2-1))

       return pd.DataFrame(rows)

def fmt(df):

       y=df.copy()

       for c in ['selected_rate','selected_base_rate','ci95_low','ci95_high','roi_at_3_20']:

           if c in y: y[c]=y[c].map(lambda v:'' if pd.isna(v) else f'{v:.2%}')

       if 'lift_vs_all_v3c_selected_pp' in y:

           y['lift_vs_all_v3c_selected_pp']=y['lift_vs_all_v3c_selected_pp'].map(lambda v:'' if pd.isna(v) else f'{v:+.2f} pp')

       if 'fair_odds_empirical' in y:

           y['fair_odds_empirical']=y['fair_odds_empirical'].map(lambda v:'' if pd.isna(v) else f'{v:.2f}')

       return y

def main():

       print('='*126)

       print("V3-C 55–65 — AUDIT EXPLORATOIRE DES TARGETS PAR SCORE À 87'")

       print('='*126)

       print('Aucun retuning. Aucun nouveau filtre de production.')

       hist=ROOT/'outputs/historical_dataset_v2/Historical_Dataset_Goal_After_87_CORE_READY_v2.csv'

       oos_path=ROOT/'outputs/grand_volume_rules_audit/rolling_oos_684_exact_predictions.csv'

       pool_path=ROOT/'outputs/v3i_unseen_holdout_500/batch2_frozen_221/V3I_BATCH2_ELIGIBLE_POOL.csv'
       b1_path=ROOT/'outputs/v3i_unseen_holdout_500/batch1_frozen_279/V3I_BATCH1_279_FROZEN.csv'
       if not b1_path.exists():
           raise FileNotFoundError(f'Fichier Batch 1 introuvable: {b1_path}')

       model_path=ROOT/'outputs/model_v3_experiments/candidates_refit_all/V3_C_AGGRESSIVE_refit_all.joblib'

       schema_path=ROOT/'outputs/model_v3_experiments/schemas/V3_C_AGGRESSIVE_schema.json'

       model=joblib.load(model_path)

       features=None

       if schema_path.exists():

           features=extract_38(json.loads(schema_path.read_text(encoding='utf-8')))

       if not features and hasattr(model,'feature_names_in_'): features=list(model.feature_names_in_)

       if not features or len(features)!=38: raise RuntimeError('Impossible de récupérer les 38 features V3-C')

       print('Features V3-C : 38/38')

       # OOS avec score/target historique

       oos=csv_read(oos_path)

       oos['p_v3c']=pd.to_numeric(oos[pcol_oos(oos)], errors='coerce')

       h=csv_read(hist)[['match_id','score_87_home','score_87_away','goal_after_87']].copy()

       oos=oos.drop(columns=[c for c in ['score_87_home','score_87_away','goal_after_87'] if c in oos.columns], errors='ignore').merge(h,on='match_id',how='left',validate='one_to_one')

       oos['target']=target_series(oos); oos['selected']=oos['p_v3c'].ge(LOW)&oos['p_v3c'].lt(HIGH); oos=add_score_fields(oos)

       # Batch1 + pool 389 scorés avec le modèle final figé

       def prep_final(d):

           X=d[features].apply(pd.to_numeric, errors='coerce')

           d=d.copy(); d['p_v3c']=model.predict_proba(X)[:,1]; d['target']=target_series(d)

           d['selected']=d['p_v3c'].ge(LOW)&d['p_v3c'].lt(HIGH)

           return add_score_fields(d)

       b1=prep_final(csv_read(b1_path))

       pool=csv_read(pool_path)

       order=pd.to_numeric(pool['_v3i_batch2_selection_order'], errors='coerce')

       b2=prep_final(pool.loc[order.between(1,221)].copy())

       r168=prep_final(pool.loc[order.between(222,389)].copy())

       samples={'OOS_684':oos,'BATCH1_279':b1,'BATCH2_221':b2,'RESERVE168_FINAL':r168}

       print('\nFIDÉLITÉ:')

       for name,d in samples.items():

           n=len(d); sn=int(d['selected'].sum()); sp=int(d.loc[d['selected'],'target'].sum())

           print(f'{name:<18} n={n:<3} | V3C55-65={sp}/{sn} = {sp/sn:.2%}')

           if (n,sn,sp)!=EXPECTED[name]: raise RuntimeError(f'Fidélité échouée pour {name}: {(n,sn,sp)} != {EXPECTED[name]}')

       ext=pd.concat([b1.assign(source='BATCH1_279'),b2.assign(source='BATCH2_221'),r168.assign(source='RESERVE168_FINAL')],ignore_index=True)

       samples['EXTERNAL668']=ext

       groups=['score_exact_87','score_state_87','margin_bucket_87','total_goals_bucket_87','score_archetype_87']

       out=[]; selected=[]

       for name,d in samples.items():

           s=d.loc[d['selected']].copy(); s['sample']=name; selected.append(s)

           for g in groups: out.append(summarize(s,name,g))

       res=pd.concat(out,ignore_index=True)

       selall=pd.concat(selected,ignore_index=True,sort=False)

       outdir=ROOT/'outputs/v3c_target_score_audit'; outdir.mkdir(parents=True,exist_ok=True)

       res.to_csv(outdir/'v3c55_65_score_target_audit_all_groupings.csv',index=False)

       for g in groups: res[res['grouping'].eq(g)].to_csv(outdir/f'v3c55_65_{g}_by_sample.csv',index=False)

       keep=[c for c in ['sample','match_id','competition_id','home_team','away_team','kickoff_utc','score_87_home','score_87_away','score_exact_87','score_state_87','margin_bucket_87','total_goals_bucket_87','score_archetype_87','p_v3c','target'] if c in selall.columns]

       selall[keep].to_csv(outdir/'v3c55_65_selected_matches_with_score_context.csv',index=False)

       for g,title in [

           ('score_exact_87',"EXTERNAL668 — SCORE EXACT À 87'"),

           ('score_state_87',"EXTERNAL668 — ÉTAT DU SCORE À 87'"),

           ('margin_bucket_87',"EXTERNAL668 — ÉCART AU SCORE À 87'"),

           ('total_goals_bucket_87',"EXTERNAL668 — TOTAL DE BUTS À 87'"),

           ('score_archetype_87',"EXTERNAL668 — ARCHÉTYPE DE SCORE À 87'"),

       ]:

           q=res[(res['sample']=='EXTERNAL668')&(res['grouping']==g)].sort_values(['selected_n','selected_rate'],ascending=[False,False])

           if g=='score_exact_87': q=q[q['selected_n']>=3]

           print('\n'+'='*126); print(title); print('='*126); print(fmt(q).to_string(index=False))

       print('\n'+'='*126)

       print('IMPORTANT')

       print('='*126)

       print("Cet audit est EXPLORATOIRE. Il peut suggérer des hypothèses par score, mais aucune règle score ne doit être ajoutée au scanner à partir de ces mêmes données.")

       print(f'Sorties: {outdir}')

       summary={k:{'n':len(v),'selected_n':int(v['selected'].sum()),'selected_positives':int(v.loc[v['selected'],'target'].sum()),'selected_rate':float(v.loc[v['selected'],'target'].mean())} for k,v in samples.items()}

       (outdir/'AUDIT_SUMMARY.json').write_text(json.dumps({'rule':'0.55 <= P(V3-C) < 0.65','status':'EXPLORATORY_SCORE_AUDIT_ONLY','samples':summary},indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':

       main()
