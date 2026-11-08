#!/usr/bin/env python3
import argparse, csv, hashlib, json, math
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

B=Path('outputs/v4_research')
G7=B/'v42_g02g7_blind12_prematch_coverage'
OUT=B/'v42_g02g8_blind12_prematch_coverage/raw_history'
TEAMS=G7/'G02G7_BLIND12_TEAMS_FROZEN.csv'
TARGETS=G7/'G02G7_BLIND12_TARGETS_FROZEN.csv'
REPORT=G7/'G02G7_PLAN_REPORT.json'
MASTER=B/'v42_g02g5_prematch_plan/G02G5_TARGET_OBSERVATIONS_FROZEN.csv'
MANIFEST=B/'v41_mass350/manifest.csv'
HASH={
 TEAMS:'d000987b8f342f5ce8f146bd79c42334848aff85b8350388958ce51a91c7cbbf',
 TARGETS:'b0fbfc697c2f992f11914068dff0b6f85551a4e89224ecb975dc06f2ee3a0cf4',
 REPORT:'037d1b0b487ce9ed8fc13450336ebc64277d0ffc44b755071e7688ac892168b9',
 MASTER:'4180cd7b325e299632658fe6cd63cb2b391dc93405e2bb90799e59adb697ccf2',
 MANIFEST:'61417a2ef02884af6a75fd2a8926961d9b59208fb0d4ce352433ecfe4eeada7e',
}

def stop(message):
 raise SystemExit('STOP : '+str(message))

def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:
  return list(csv.DictReader(f))

def utc(v):
 try:
  d=datetime.fromisoformat(v.replace('Z','+00:00'))
  if d.tzinfo is None:
   raise ValueError('fuseau manquant')
  return d.astimezone(timezone.utc)
 except (AttributeError, ValueError) as e:
  stop('date incorrecte '+repr(v)+' '+str(e))

def valid_score(v):
 if v is None or isinstance(v,bool):
  return False
 try:
  x=float(v)
  return math.isfinite(x) and x>=0 and x.is_integer()
 except (TypeError,ValueError):
  return False

def check_page(wrapper,tid):
 if wrapper.get('team_id')!=tid or wrapper.get('endpoint')!='/football/matches':
  stop('enveloppe incorrecte '+tid)

 p=wrapper.get('params',{})
 if not isinstance(p,dict) or (
  p.get('team_id'),p.get('status'),
  p.get('per_page'),p.get('page',1)
 )!=(tid,'finished',100,1):
  stop('parametres incorrects '+tid)

 r=wrapper.get('response')
 if not isinstance(r,dict) or not isinstance(r.get('meta'),dict) or not isinstance(r.get('data'),list) or r.get('error'):
  stop('reponse API incorrecte '+tid)

 meta,items=r['meta'],r['data']

 try:
  n,total,pages,limit=(
   int(meta[k])
   for k in ('page','total','total_pages','per_page')
  )
 except (KeyError,ValueError,TypeError):
  stop('metadonnees incompletes '+tid)

 if n!=1 or limit!=100 or pages<1 or len(items)>100 or total<len(items):
  stop('pagination incorrecte '+tid)

 dated=[]
 for m in items:
  if not isinstance(m,dict) or not m.get('id') or m.get('status')!='finished':
   stop('match invalide '+tid)

  h,a=m.get('home_team'),m.get('away_team')
  if not isinstance(h,dict) or not isinstance(a,dict) or (
   h.get('id')==tid
  )+(
   a.get('id')==tid
  )!=1:
   stop('equipes invalides '+str(m.get('id')))

  dated.append((utc(m.get('utc_date')),m['id'],m))

 if len({mid for _,mid,_ in dated})!=len(dated):
  stop('doublons '+tid)

 if any(
  dated[i][0]<dated[i+1][0]
  for i in range(len(dated)-1)
 ):
  stop('ordre des rencontres non decroissant '+tid)

 return dated,meta

ap=argparse.ArgumentParser()
ap.add_argument('--budget',type=int,default=0)
args=ap.parse_args()

if not 0<=args.budget<=12:
 stop('budget entre 0 et 12')

print('=== G02-G8 | PILOTE AVEUGLE | budget =',args.budget,'===')

for path,digest in HASH.items():
 if not path.is_file() or sha(path)!=digest:
  stop('SHA invalide '+str(path))
 print('SHA OK :',path.name)

team_rows=read(TEAMS)
obs=read(TARGETS)
all_obs=read(MASTER)
manifest_rows=read(MANIFEST)
manifest={r['match_id']:r for r in manifest_rows}

if (
 len(team_rows),len(obs),len(all_obs),
 len(manifest_rows),len(manifest)
)!=(12,30,3000,12415,12415):
 stop('effectifs incoherents')

ids=[r['team_id'] for r in team_rows]

if len(set(ids))!=12 or Counter(
 r['sample_stratum'] for r in team_rows
)!=Counter({'SINGLE':4,'REPEATED':4,'FREQUENT':4}):
 stop('selection non conforme')

report=json.loads(REPORT.read_text(encoding='utf-8'))
if sorted(report.get('team_ids',[]))!=sorted(ids):
 stop('rapport different')

master_keys={
 (r['match_id'],r['side'],r['team_id'])
 for r in all_obs
}

if len(master_keys)!=3000:
 stop('observations master dupliquees')

by=defaultdict(list)

for o in obs:
 mid,side,tid=o['match_id'],o['side'],o['team_id']

 if tid not in ids or mid not in manifest or side not in ('home','away'):
  stop('observation incorrecte '+mid)

 if (mid,side,tid) not in master_keys or manifest[mid][side+'_id']!=tid:
  stop('jointure incorrecte '+mid)

 if utc(manifest[mid]['kickoff_utc'])-utc(o['safe_cutoff_utc'])!=timedelta(hours=4):
  stop('cutoff incorrect '+mid)

 by[tid].append(o)

for row in team_rows:
 if len(by[row['team_id']])!=int(row['target_observations']):
  stop('comptage incorrect')

cache={}
missing=[]

for tid in ids:
 path=OUT/(tid+'.json')

 if path.exists():
  wrapper=json.loads(path.read_text(encoding='utf-8'))
  dated,meta=check_page(wrapper,tid)
  cache[tid]=(dated,meta)
  print('CACHE OK',tid,'SHA256',sha(path))
 else:
  missing.append(tid)
  print('CACHE ABSENT',tid,'| observations',len(by[tid]))

print('Equipes : 12 | observations : 30 | cache manquants :',len(missing))

if args.budget==0:
 print('G02-G8 PLAN OK | 0 APPEL API | 0 ECRITURE | 0 ENTRAINEMENT')
 raise SystemExit(0)

if missing:
 import rebuild_clean_prematch_v2 as pre

 lab=pre.load_lab()
 client=pre.build_rate_limited_client(
  lab,max_retries=1,min_interval=2.5
 )

 for i,tid in enumerate(missing[:args.budget],1):
  params={
   'team_id':tid,
   'status':'finished',
   'per_page':100,
   'page':1
  }

  print('\nAPPEL',i,'/',args.budget,tid)
  response=client.get('/football/matches',params=params)

  wrapper={
   'team_id':tid,
   'endpoint':'/football/matches',
   'params':params,
   'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
   'status':'RAW_NOT_CERTIFIED',
   'response':response
  }

  dated,meta=check_page(wrapper,tid)

  OUT.mkdir(parents=True,exist_ok=True)
  dest=OUT/(tid+'.json')
  temp=OUT/(tid+'.json.tmp')

  if dest.exists() or temp.exists():
   stop('refus decrasement '+str(dest))

  with temp.open('x',encoding='utf-8') as f:
   json.dump(wrapper,f,ensure_ascii=False,indent=2)

  temp.replace(dest)
  cache[tid]=(dated,meta)

  print(
   'SAUVE',dest,
   '| matchs',len(dated),
   '| total',meta['total'],
   '| pages',meta['total_pages'],
   '| SHA256',sha(dest)
  )

print('\n=== CONTROLE DE COUVERTURE DES CACHES ===')
c=Counter()

for tid in ids:
 if tid not in cache:
  print('EN ATTENTE',tid)
  continue

 dated,meta=cache[tid]

 for o in sorted(
  by[tid],
  key=lambda x:(x['safe_cutoff_utc'],x['match_id'])
 ):
  mid=o['match_id']
  cutoff=utc(o['safe_cutoff_utc'])

  prior=[
   e for e in dated
   if e[0]<cutoff and e[1]!=mid
  ]

  top=prior[:20]
  scored=0

  for dt,hid,m in top:
   side='home' if m['home_team']['id']==tid else 'away'
   score=m.get('score')
   reg=score.get('regulation') if isinstance(score,dict) else None
   scored+=valid_score(
    reg.get(side) if isinstance(reg,dict) else None
   )

  lead=(
   (cutoff-top[0][0]).total_seconds()/3600
   if top else None
  )

  c['observations']+=1
  c['has20']+=len(top)==20
  c['complete_scores20']+=len(top)==20 and scored==20
  c['recent_match_start_under24h']+=lead is not None and lead<24

  print(
   tid,mid,o['side'],
   '| historiques avant cutoff',len(prior),
   '| selection',len(top),
   '| scores',scored,
   '| derniere rencontre commencee depuis (h)',
   round(lead,2) if lead is not None else None,
   '| pagination',meta['page'],'/',meta['total_pages']
  )

print('BILAN :',dict(c),'| equipes en cache',len(cache),'/12')
print('ATTENTION : statut finished au jour de collecte != certification historique as-of.')
print('AUCUN LABEL LU. AUCUNE FEATURE CERTIFIEE. AUCUN ENTRAINEMENT.')
print('G02-G8 TERMINE')
