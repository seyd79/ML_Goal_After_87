#!/usr/bin/env python3
import csv, hashlib, json, argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

B=Path('outputs/v4_research')
T=B/'v42_g02g5_prematch_plan/G02G5_TARGET_OBSERVATIONS_FROZEN.csv'
U=B/'v42_g02g5_prematch_plan/G02G5_TEAMS_FROZEN.csv'
M=B/'v41_mass350/manifest.csv'
P=B/'v42_g02g6_prematch_pilot/G02G6_PILOT6_TEAMS_FROZEN.csv'
OUT=B/'v42_g02g7_blind12_prematch_coverage'
SHA={
 T:'4180cd7b325e299632658fe6cd63cb2b391dc93405e2bb90799e59adb697ccf2',
 U:'1b31c5763e6aa2c0fd27d1699d83c6ce1db72a4a0431d6d5cd4a7ce16aff0bb1',
 M:'61417a2ef02884af6a75fd2a8926961d9b59208fb0d4ce352433ecfe4eeada7e',
 P:'a136f915f5cc87106d085d162660eb699eb2282f904929ef9cc21067471f24a0',
}
SEED='G02G7|blind12|fixed20|2026-11-03|v1'

def stop(s):
 raise SystemExit('STOP : '+s)

def digest(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()

def load(p):
 with p.open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f)
  return list(rd),rd.fieldnames

def utc(s):
 try:
  d=datetime.fromisoformat(s.replace('Z','+00:00'))
  if d.tzinfo is None:
   raise ValueError('timezone absent')
  return d.astimezone(timezone.utc)
 except (ValueError,AttributeError):
  stop('datetime invalide : '+repr(s))

def write_new(p,fields,rows):
 if p.exists():
  stop('refus ecrasement '+str(p))
 with p.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
  w.writeheader()
  w.writerows(rows)
 print('GEL :',p,'SHA256 :',digest(p))

arg=argparse.ArgumentParser()
arg.add_argument('--freeze',action='store_true')
a=arg.parse_args()

print('='*68)
print('G02-G7 | SELECTION AVEUGLE DE 12 EQUIPES')
print('='*68)

for p,h in SHA.items():
 if not p.is_file() or digest(p)!=h:
  stop('source absente ou modifiee '+str(p))
 print('SHA OK :',p)

obs,of=load(T)
teams,tf=load(U)
manifest,_=load(M)
old,_=load(P)

if (len(obs),len(teams),len(manifest),len(old))!=(3000,1116,12415,6):
 stop('effectifs inattendus')

if len({r['match_id'] for r in manifest})!=12415:
 stop('manifest duplique')
if len({r['team_id'] for r in teams})!=1116:
 stop('equipes dupliquees')

md={r['match_id']:r for r in manifest}
by=defaultdict(list)
seen=set()

for r in obs:
 mid,side,tid=r['match_id'],r['side'],r['team_id']

 if (mid,side) in seen or mid not in md or side not in ('home','away'):
  stop('observation invalide '+mid)

 seen.add((mid,side))
 m=md[mid]

 if m[side+'_id']!=tid or utc(r['kickoff_utc'])!=utc(m['kickoff_utc']):
  stop('identite ou date invalide '+mid)

 if utc(m['kickoff_utc'])-utc(r['safe_cutoff_utc'])!=timedelta(hours=4):
  stop('cutoff invalide '+mid)

 by[tid].append(r)

if len(seen)!=3000 or len(by)!=1116:
 stop('jointures inattendues')

for r in teams:
 tid=r['team_id']

 if len(by[tid])!=int(r['target_observations']):
  stop('nombre incorrect '+tid)

 if utc(r['earliest_cutoff_utc'])!=min(
  utc(x['safe_cutoff_utc']) for x in by[tid]
 ):
  stop('cutoff minimum incorrect '+tid)

 if utc(r['latest_cutoff_utc'])!=max(
  utc(x['safe_cutoff_utc']) for x in by[tid]
 ):
  stop('cutoff maximum incorrect '+tid)

oldids={r['team_id'] for r in old}

if len(oldids)!=6 or not oldids.issubset(by):
 stop('ancien pilote incorrect')

groups=defaultdict(list)

def group(n):
 return 'SINGLE' if n==1 else ('REPEATED' if n<=3 else 'FREQUENT')

for tid,lst in by.items():
 if tid not in oldids:
  groups[group(len(lst))].append(tid)

chosen={}
names=('SINGLE','REPEATED','FREQUENT')

for name in names:
 if len(groups[name])<4:
  stop('moins de quatre candidats '+name)

 ranked=sorted(
  groups[name],
  key=lambda t:(
   hashlib.sha256((SEED+'|'+name+'|'+t).encode()).hexdigest(),
   t
  )
 )

 for tid in ranked[:4]:
  chosen[tid]=name

result_teams=[]

for r in teams:
 tid=r['team_id']
 if tid in chosen:
  result_teams.append(
   dict(r,sample_stratum=chosen[tid],sample_seed=SEED)
  )

result_teams.sort(
 key=lambda r:(names.index(r['sample_stratum']),r['team_id'])
)

result_obs=sorted(
 (r for r in obs if r['team_id'] in chosen),
 key=lambda r:(
  r['team_id'],r['safe_cutoff_utc'],
  r['match_id'],r['side']
 )
)

print('\nGroupes disponibles hors ancien pilote :')
print({g:len(groups[g]) for g in names})

for r in result_teams:
 print(
  r['team_id'],'|',r['sample_stratum'],
  '| observations :',r['target_observations'],
  '| dates :',r['earliest_cutoff_utc'],
  '--',r['latest_cutoff_utc']
 )

print(
 '12 equipes |',len(result_obs),
 'observations | mois :',
 dict(sorted(Counter(
  r['safe_cutoff_utc'][:7] for r in result_obs
 ).items()))
)
print('Aucun label lu. Aucune API. Aucun entrainement.')

if a.freeze:
 paths=(
  OUT/'G02G7_BLIND12_TEAMS_FROZEN.csv',
  OUT/'G02G7_BLIND12_TARGETS_FROZEN.csv',
  OUT/'G02G7_PLAN_REPORT.json'
 )

 if any(p.exists() for p in paths):
  stop('sortie deja existante')

 OUT.mkdir(parents=True,exist_ok=True)
 write_new(
  paths[0],tf+['sample_stratum','sample_seed'],result_teams
 )
 write_new(paths[1],of,result_obs)

 report={
  'stage':'G02-G7',
  'selection_seed':SEED,
  'team_ids':sorted(chosen),
  'observations':len(result_obs),
  'sources_sha256':{str(p):v for p,v in SHA.items()},
  'no_labels_used':True,
  'api_calls':0,
  'training_allowed':False,
  'prematch_contract_candidate':{
   'last_n_matches':20,
   'cutoff':'target kickoff minus 4h',
   'score_field':'score.regulation',
   'fewer_than_20':'MISSING',
   'historical_result_asof_certified':False
  }
 }

 with paths[2].open('x',encoding='utf-8') as f:
  json.dump(report,f,indent=2,ensure_ascii=False)

 print('GEL :',paths[2],'SHA256 :',digest(paths[2]))
 print('G02-G7 FREEZE OK')

else:
 print('G02-G7 PLAN OK — AUCUNE ECRITURE')
