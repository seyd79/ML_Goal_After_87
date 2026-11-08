#!/usr/bin/env python3
import argparse, csv, hashlib, json, math
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

B=Path('outputs/v4_research')
G5=B/'v42_g02g5_prematch_plan'
G6=B/'v42_g02g6_prematch_pilot'
G7=B/'v42_g02g7_blind12_prematch_coverage'
G8=B/'v42_g02g8_blind12_prematch_coverage'
OUT=B/'v42_g02g10_prematch_contract'
T=G5/'G02G5_TARGET_OBSERVATIONS_FROZEN.csv'
M=B/'v41_mass350/manifest.csv'
P6=G6/'G02G6_PILOT6_TEAMS_FROZEN.csv'
P12=G7/'G02G7_BLIND12_TEAMS_FROZEN.csv'
T12=G7/'G02G7_BLIND12_TARGETS_FROZEN.csv'
SHA={
 T:'4180cd7b325e299632658fe6cd63cb2b391dc93405e2bb90799e59adb697ccf2',
 M:'61417a2ef02884af6a75fd2a8926961d9b59208fb0d4ce352433ecfe4eeada7e',
 P6:'a136f915f5cc87106d085d162660eb699eb2282f904929ef9cc21067471f24a0',
 P12:'d000987b8f342f5ce8f146bd79c42334848aff85b8350388958ce51a91c7cbbf',
 T12:'b0fbfc697c2f992f11914068dff0b6f85551a4e89224ecb975dc06f2ee3a0cf4'
}
CACHE_SHA={
 'tm_033281':'c8619b3e29dbafd021246d2ee9421a36ae728f76cce44dd3e279092b8fb8c120',
 'tm_0845345':'a502a4cfb712d149f6af0312f0261d81dce4bad56d5d97d7d7c3f7cf480ab482',
 'tm_087341':'1693cd6bf92bf704c2317f90ac40079d217ed72cfc3bc95b0f50803dd9599c87',
 'tm_15775':'2a9dde5fa73199ac68735768b34f283a01ecebb4a6f84551d914a760da638dfd',
 'tm_1699489':'97e99fc5a0b6f69c0272f2d02a7de6ced7f52a2863e5be06f6db5ceb42fc6755',
 'tm_19704':'7f04081190aca47d1a817fa0ad11cd83602433b2cf6dee08d6b423b216289b73',
 'tm_279303':'40561207f351e74bd6e3dff22f77df04c86d5983c1ffc7bd7d68c3e5277e2c12',
 'tm_32821':'41f44b3a6746b81edd73389a0cf125bf5dfad04e540536df061f9bcb1f887635',
 'tm_34030':'dc5a24b105aed591cdc3a65e6ea760602f50826346af3205a9a8e59b93124367',
 'tm_343130':'c92b22fe55381a206073148b52758d905f33af1ef2b788a0ad0754b71b281cbe',
 'tm_37445':'c3900ea631f71c7f65405137c0b6f67d1f503f6774a6f6e29f5959671c8d7daf',
 'tm_46516':'29b25d63a5bc29cc510abb843aa97143d923855fc102ceaa909f18ee6d48d15d',
 'tm_4748608':'7e65c2a35fd84588e49afa214d86344112f69ded186b0848e92a2ac284cf38d0',
 'tm_6047':'adef09c91a4f33bf8c846414dc7a5b6591610c2f39100e8f8d841efe6e595c67',
 'tm_6637':'cafbb5438644bac78c215ef39ed0a97fc60c5de50dba63d021b24ee7dde53af3',
 'tm_7243233':'0def1c56ea970aa793dce42da62dcc7353399dccb7ed989c5eeace9433aa0e97',
 'tm_85960':'cf0e61dee64ae65c7dd0135ddb90c1974ad1314e92a75451afa13200a3a30e9d',
 'tm_9715':'bb1ee6bf16b0e4e15e3a77249f3980223cbbeecdca91d65975ad6ab037afeebc',
}
def stop(s): raise SystemExit('STOP : '+s)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def utc(s):
 d=datetime.fromisoformat(s.replace('Z','+00:00'))
 if d.tzinfo is None: stop('fuseau absent '+s)
 return d.astimezone(timezone.utc)
def goals(v):
 if v is None or isinstance(v,bool): return None
 try:
  x=float(v)
  return int(x) if math.isfinite(x) and x>=0 and x.is_integer() else None
 except (ValueError,TypeError): return None
p=argparse.ArgumentParser(); p.add_argument('--freeze',action='store_true'); a=p.parse_args()
print('=== G02-G10 | CONTRAT PREMATCH 20 MATCHS | PILOTE 52 ===')
for path,h in SHA.items():
 if not path.exists() or sha(path)!=h: stop('SHA source '+str(path))
old={r['team_id'] for r in rows(P6)}; new={r['team_id'] for r in rows(P12)}
if len(old)!=6 or len(new)!=12 or old&new or old|new!=set(CACHE_SHA): stop('selection pilote invalide')
allobs=rows(T); man=rows(M)
if len(allobs)!=3000 or len(man)!=12415: stop('effectifs source invalides')
manifest={r['match_id']:r for r in man}
obs=[r for r in allobs if r['team_id'] in old|new]
if len(obs)!=52 or len(manifest)!=12415: stop('effectifs jointure invalides')
ref12={(r['match_id'],r['side'],r['team_id']) for r in rows(T12)}
if len(ref12)!=30 or ref12!={(r['match_id'],r['side'],r['team_id']) for r in obs if r['team_id'] in new}:
 stop('jointure G7 invalide')
history={}; cache_sha={}
for team in sorted(old|new):
 path=(G6 if team in old else G8)/'raw_history'/(team+'.json')
 if not path.is_file() or sha(path)!=CACHE_SHA[team]: stop('cache absent ou SHA different '+team)
 wrapper=json.loads(path.read_text(encoding='utf-8'))
 if wrapper.get('team_id')!=team: stop('cache equipe invalide '+team)
 response=wrapper.get('response',{}); meta=response.get('meta',{}); items=response.get('data')
 if meta.get('page')!=1 or not isinstance(items,list) or len(items)>100: stop('page invalide '+team)
 page=[]
 for item in items:
  if item.get('status')!='finished': stop('statut non fini '+team)
  h=item.get('home_team') or {}; away=item.get('away_team') or {}
  if (h.get('id')==team)+(away.get('id')==team)!=1: stop('identite equipe '+team)
  page.append((utc(item['utc_date']),item['id'],item))
 if len({i for _,i,_ in page})!=len(page): stop('doublons '+team)
 if any(page[j][0]<page[j+1][0] for j in range(len(page)-1)):
  stop('historique non decroissant '+team)
 history[team]=page; cache_sha[team]=sha(path)
counts=Counter(); output=[]; leads=[]
for r in sorted(obs,key=lambda x:(x['team_id'],x['safe_cutoff_utc'],x['match_id'])):
 team,mid,side=r['team_id'],r['match_id'],r['side']
 if mid not in manifest or manifest[mid][side+'_id']!=team: stop('jointure match '+mid)
 cutoff=utc(r['safe_cutoff_utc'])
 if utc(manifest[mid]['kickoff_utc'])-cutoff!=timedelta(hours=4): stop('cutoff '+mid)
 selected=[(d,i,m) for d,i,m in history[team] if d<cutoff and i!=mid][:20]
 values=[]
 for d,i,item in selected:
  team_side='home' if item['home_team']['id']==team else 'away'
  regulation=(item.get('score') or {}).get('regulation')
  val=goals(regulation.get(team_side)) if isinstance(regulation,dict) else None
  if val is None: stop('score invalide : '+i)
  values.append(val)
 n=len(values)
 status='COMPLETE20' if n==20 else ('PARTIAL10_19' if n>=10 else 'MISSING_LT10')
 avg=sum(values)/n if n>=10 else None
 age=(cutoff-selected[0][0]).total_seconds()/3600 if selected else None
 counts[status]+=1; counts['score_uses']+=n
 if age is not None: leads.append(age)
 output.append({'match_id':mid,'side':side,'team_id':team,'safe_cutoff_utc':r['safe_cutoff_utc'],
 'history_n':n,'status':status,'avg_goals_scored':f'{avg:.12f}' if avg is not None else '',
 'latest_match_lead_hours':f'{age:.2f}' if age is not None else '',
 'asof_certified':'0','source_sha256':cache_sha[team]})
print('Equipes :',len(history),'| observations :',len(output))
print('Couverture :',dict(counts))
print('Age derniere rencontre min/max :',round(min(leads),2),'h /',round(max(leads)/24,2),'jours')
print('Historique incomplet :',[(r['team_id'],r['match_id'],r['history_n']) for r in output if r['status']!='COMPLETE20'])
if len(output)!=52 or counts['COMPLETE20']!=51 or counts['PARTIAL10_19']!=1 or counts['score_uses']!=1039:
 stop('bilan G02-G9 different')
contract={
 'version':'G02-G10-V1-CANDIDATE',
 'feature_names':['home_avg_goals_scored','away_avg_goals_scored'],
 'historical_window':'up to 20 latest finished matches with kickoff UTC strictly before cutoff',
 'cutoff':'target kickoff UTC minus 4h',
 'goals_source':'score.regulation, side matched by team_id',
 'min_history':10,
 'history_10_to_19':'mean over actual n; history_n tracked as metadata',
 'history_0_to_9':'missing; train-only imputation after time split',
 'history_20_plus':'mean over latest 20 only',
 'status':'RESEARCH_ONLY_NOT_ASOF_CERTIFIED',
 'future_matches_allowed':False,
 'training_approved':False,
 'hashes_frozen_inputs':{str(k):v for k,v in SHA.items()},
 'hashes_cached_history':cache_sha,
}
if not a.freeze:
 print('G02-G10 PLAN OK | AUCUNE API | AUCUNE ECRITURE | AUCUN ENTRAINEMENT')
else:
 files=[OUT/'G02G10_CONTRACT_FROZEN.json',OUT/'G02G10_PILOT52_RESEARCH.csv']
 if any(p.exists() for p in files): stop('sortie deja existante')
 OUT.mkdir(parents=True,exist_ok=True)
 with files[0].open('x',encoding='utf-8') as f: json.dump(contract,f,indent=2,ensure_ascii=False)
 with files[1].open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
 for p in files: print('GEL',p,'SHA256',sha(p))
 print('G02-G10 FREEZE OK | AS-OF NON CERTIFIE | AUCUN ENTRAINEMENT')
