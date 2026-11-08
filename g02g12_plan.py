#!/usr/bin/env python3
import argparse, csv, hashlib, json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

B = Path('outputs/v4_research')
G5 = B/'v42_g02g5_prematch_plan'
G6 = B/'v42_g02g6_prematch_pilot'
G7 = B/'v42_g02g7_blind12_prematch_coverage'
G11 = B/'v42_g02g11_prematch_contract'
OUT = B/'v42_g02g12_remaining_prematch_plan'

T = G5/'G02G5_TARGET_OBSERVATIONS_FROZEN.csv'
U = G5/'G02G5_TEAMS_FROZEN.csv'
M = B/'v41_mass350/manifest.csv'
P6 = G6/'G02G6_PILOT6_TEAMS_FROZEN.csv'
P12 = G7/'G02G7_BLIND12_TEAMS_FROZEN.csv'
O12 = G7/'G02G7_BLIND12_TARGETS_FROZEN.csv'
C = G11/'G02G11_PREMATCH_CONTRACT_FROZEN.json'

EXPECTED = {
 T:'4180cd7b325e299632658fe6cd63cb2b391dc93405e2bb90799e59adb697ccf2',
 U:'1b31c5763e6aa2c0fd27d1699d83c6ce1db72a4a0431d6d5cd4a7ce16aff0bb1',
 M:'61417a2ef02884af6a75fd2a8926961d9b59208fb0d4ce352433ecfe4eeada7e',
 P6:'a136f915f5cc87106d085d162660eb699eb2282f904929ef9cc21067471f24a0',
 P12:'d000987b8f342f5ce8f146bd79c42334848aff85b8350388958ce51a91c7cbbf',
 O12:'b0fbfc697c2f992f11914068dff0b6f85551a4e89224ecb975dc06f2ee3a0cf4',
 C:'07ff1e2c3191008861134ee5ab6eabcd76b7a44c05995d5273df4518c5e89984',
}

SEED = 'G02G12|prematch_remaining|blind|v1|2026-11-04'
STRATA = ('SINGLE','REPEATED','FREQUENT')

def stop(s):
 raise SystemExit('STOP : '+str(s))

def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:
  obj=csv.DictReader(f)
  return list(obj),list(obj.fieldnames or [])

def utc(v):
 try:
  d=datetime.fromisoformat(str(v).replace('Z','+00:00'))
  if d.tzinfo is None:
   raise ValueError()
  return d.astimezone(timezone.utc)
 except (ValueError,TypeError):
  stop('date invalide : '+repr(v))

def group(n):
 return 'SINGLE' if n==1 else (
  'REPEATED' if n<=3 else 'FREQUENT'
 )

def write(p,fields,rows):
 with p.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(
   f,fieldnames=fields,lineterminator='\n'
  )
  w.writeheader()
  w.writerows(rows)
 print('GEL :',p,'| SHA256 :',sha(p))

ap=argparse.ArgumentParser()
ap.add_argument('--freeze',action='store_true')
args=ap.parse_args()

print('='*75)
print('G02-G12 | PLAN DES EQUIPES PREMATCH RESTANTES')
print('='*75)

for p,h in EXPECTED.items():
 if not p.is_file() or sha(p)!=h:
  stop('SHA non conforme : '+str(p))
 print('SHA OK :',p.name)

contract=json.loads(C.read_text(encoding='utf-8'))

if (
 contract.get('id')!='G02G11_PREMATCH_V1'
 or contract.get('last_n_historical_matches')!=20
 or contract.get('historical_status_asof_certified') is not False
 or contract.get('training_allowed') is not False
):
 stop('contrat inattendu')

obs,of=read(T)
teams,tf=read(U)
man,_=read(M)
pilot6,_=read(P6)
pilot12,_=read(P12)
targets12,_=read(O12)

if tuple(map(
 len,(obs,teams,man,pilot6,pilot12,targets12)
))!=(3000,1116,12415,6,12,30):
 stop('effectifs sources inattendus')

md={r['match_id']:r for r in man}
td={r['team_id']:r for r in teams}

if len(md)!=12415 or len(td)!=1116:
 stop('IDs sources dupliques')

old=(
 {r['team_id'] for r in pilot6}
 | {r['team_id'] for r in pilot12}
)

if len(old)!=18 or not old.issubset(td):
 stop('18 equipes pilotes incorrectes')

by=defaultdict(list)
seen=set()
sides=defaultdict(set)

for r in obs:
 mid,side,tid=r['match_id'],r['side'],r['team_id']

 if (
  (mid,side) in seen
  or mid not in md
  or side not in ('home','away')
 ):
  stop('observation invalide : '+mid)

 seen.add((mid,side))
 sides[mid].add(side)

 if (
  md[mid][side+'_id']!=tid
  or utc(md[mid]['kickoff_utc'])!=utc(r['kickoff_utc'])
  or utc(r['kickoff_utc'])-utc(r['safe_cutoff_utc'])
     !=timedelta(hours=4)
 ):
  stop('equipe/date/cutoff invalide : '+mid)

 by[tid].append(r)

if (
 len(seen)!=3000
 or len(by)!=1116
 or len(sides)!=1500
 or any(v!={'home','away'} for v in sides.values())
):
 stop('3000 observations home/away non conformes')

for tid,t in td.items():
 entries=by[tid]

 if (
  len(entries)!=int(t['target_observations'])
  or min(utc(x['safe_cutoff_utc']) for x in entries)
     !=utc(t['earliest_cutoff_utc'])
  or max(utc(x['safe_cutoff_utc']) for x in entries)
     !=utc(t['latest_cutoff_utc'])
 ):
  stop('statistiques equipe divergentes : '+tid)

pilot_obs=sum(len(by[t]) for t in old)

if pilot_obs!=52:
 stop('52 observations pilotes attendues')

ref12={
 (r['match_id'],r['side'],r['team_id'])
 for r in targets12
}
pilot12ids={x['team_id'] for x in pilot12}
cur12={
 (r['match_id'],r['side'],r['team_id'])
 for r in obs if r['team_id'] in pilot12ids
}

if len(ref12)!=30 or ref12!=cur12:
 stop('pilot 12 different')

remaining=set(td)-old

if (
 len(remaining)!=1098
 or sum(len(by[tid]) for tid in remaining)!=2948
):
 stop('1098 equipes / 2948 observations attendues')

buckets={}
for name in STRATA:
 ids=[
  tid for tid in remaining
  if group(len(by[tid]))==name
 ]
 buckets[name]=sorted(
  ids,
  key=lambda tid:(
   hashlib.sha256(
    (SEED+'|'+name+'|'+tid).encode()
   ).hexdigest(),tid
  )
 )

queue=[]
for i in range(max(map(len,buckets.values()))):
 for name in STRATA:
  if i<len(buckets[name]):
   queue.append((buckets[name][i],name))

if len({tid for tid,_ in queue})!=1098:
 stop('file de collecte incoherente')

rows=[]
targets=[]

for rank,(tid,name) in enumerate(queue,1):
 rows.append({
  'queue_rank':rank,
  'stratum':name,
  **td[tid]
 })

 for x in sorted(
  by[tid],
  key=lambda z:(
   z['safe_cutoff_utc'],z['match_id'],z['side']
  )
 ):
  targets.append({'queue_rank':rank,**x})

print('\nEquipes totales :',len(teams))
print('Observations totales :',len(obs))
print('Pilotes :',len(old),'equipes /',pilot_obs,'observations')
print('Restantes :',len(queue),'equipes /',len(targets),'observations')
print('Repartition :',{
 n:len(buckets[n]) for n in STRATA
})

print('\n=== 12 PREMIERES EQUIPES DE LA FILE ===')
for r in rows[:12]:
 print(
  'rang',r['queue_rank'],
  '|',r['team_id'],
  '|',r['stratum'],
  '| obs',r['target_observations']
 )

if not args.freeze:
 print('\nG02-G12 PLAN OK')
 print('0 API | 0 ECRITURE | AUCUN LABEL | AUCUN ENTRAINEMENT')
else:
 files=[
  OUT/'G02G12_REMAINING_TEAMS_FROZEN.csv',
  OUT/'G02G12_REMAINING_TARGETS_FROZEN.csv',
  OUT/'G02G12_PLAN_REPORT.json'
 ]

 if any(p.exists() for p in files):
  stop('sortie deja presente, refus ecrasement')

 OUT.mkdir(parents=True,exist_ok=True)

 write(files[0],['queue_rank','stratum']+tf,rows)
 write(files[1],['queue_rank']+of,targets)

 report={
  'stage':'G02-G12',
  'contract_sha256':EXPECTED[C],
  'seed':SEED,
  'remaining_teams':1098,
  'remaining_observations':2948,
  'strata':{
   n:len(buckets[n]) for n in STRATA
  },
  'source_sha256':{
   str(p):h for p,h in EXPECTED.items()
  },
  'teams_queue_sha256':sha(files[0]),
  'targets_queue_sha256':sha(files[1]),
  'api_calls':0,
  'used_labels':False,
  'training_allowed':False,
  'asof_certified':False
 }

 with files[2].open('x',encoding='utf-8') as f:
  json.dump(report,f,indent=2,sort_keys=True)
  f.write('\n')

 print('GEL :',files[2],'| SHA256 :',sha(files[2]))
 print('\nG02-G12 FREEZE OK')
 print('0 API | AUCUN LABEL | AUCUN ENTRAINEMENT')
