import csv, hashlib, json, subprocess, sys
from pathlib import Path

B=Path('outputs/v4_research')
G6=B/'v42_g02g6_prematch_pilot'
G7=B/'v42_g02g7_blind12_prematch_coverage'
G8=B/'v42_g02g8_blind12_prematch_coverage'
OUT=B/'v42_g02g11_prematch_contract/G02G11_PREMATCH_CONTRACT_FROZEN.json'

SOURCES={
 'outputs/v4_research/v42_g02g5_prematch_plan/G02G5_TARGET_OBSERVATIONS_FROZEN.csv':'4180cd7b325e299632658fe6cd63cb2b391dc93405e2bb90799e59adb697ccf2',
 'outputs/v4_research/v42_g02g5_prematch_plan/G02G5_TEAMS_FROZEN.csv':'1b31c5763e6aa2c0fd27d1699d83c6ce1db72a4a0431d6d5cd4a7ce16aff0bb1',
 'outputs/v4_research/v41_mass350/manifest.csv':'61417a2ef02884af6a75fd2a8926961d9b59208fb0d4ce352433ecfe4eeada7e',
 str(G6/'G02G6_PILOT6_TEAMS_FROZEN.csv'):'a136f915f5cc87106d085d162660eb699eb2282f904929ef9cc21067471f24a0',
 str(G7/'G02G7_BLIND12_TEAMS_FROZEN.csv'):'d000987b8f342f5ce8f146bd79c42334848aff85b8350388958ce51a91c7cbbf',
 str(G7/'G02G7_BLIND12_TARGETS_FROZEN.csv'):'b0fbfc697c2f992f11914068dff0b6f85551a4e89224ecb975dc06f2ee3a0cf4',
 str(G7/'G02G7_PLAN_REPORT.json'):'037d1b0b487ce9ed8fc13450336ebc64277d0ffc44b755071e7688ac892168b9',
 str(G6/'raw_history_pages/tm_279303_page_2.json'):'6a65a6dc905e7ab9cd51781dc2d933d20b0828771a56f91ab8c3e872078424f9',
}

CACHES={
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

def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()

def stop(msg):
 raise SystemExit('STOP : '+msg)

def csvrows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:
  return list(csv.DictReader(f))

if OUT.exists():
 stop('contrat deja gele, aucun ecrasement')

script=Path('g02g10_contract.py')
if not script.is_file():
 stop('script G02-G10 absent')

for name,expected in SOURCES.items():
 p=Path(name)
 if not p.is_file() or sha(p)!=expected:
  stop('empreinte incorrecte : '+name)

old={r['team_id'] for r in csvrows(G6/'G02G6_PILOT6_TEAMS_FROZEN.csv')}
new={r['team_id'] for r in csvrows(G7/'G02G7_BLIND12_TEAMS_FROZEN.csv')}

if len(old)!=6 or len(new)!=12 or old&new or old|new!=set(CACHES):
 stop('selection equipes incoherente')

cache_paths={}
for tid,expected in CACHES.items():
 p=(G6 if tid in old else G8)/'raw_history'/(tid+'.json')
 if not p.is_file() or sha(p)!=expected:
  stop('cache non conforme : '+tid)
 cache_paths[str(p)]=expected

run=subprocess.run(
 [sys.executable,str(script)],
 capture_output=True,text=True
)

required=[
 'G02-G10 PLAN OK',
 'Equipes : 18',
 'Observations : 52',
 'Completes (20) : 51',
 'Partielles (10-19) : 1',
 'Insuffisantes (<10) : 0',
 'Scores utilises : 1039',
]

if run.returncode or any(x not in run.stdout for x in required):
 print(run.stdout[-2500:])
 print(run.stderr[-500:])
 stop('recontrole G02-G10 non conforme')

print('G02-G10 RECONTROLE : PASS | 18 equipes | 52 observations | 51+1')

contract={
 'id':'G02G11_PREMATCH_V1',
 'status':'METHODOLOGY_FROZEN_RESEARCH_ONLY',
 'feature_names':['home_avg_goals_scored','away_avg_goals_scored'],
 'last_n_historical_matches':20,
 'cutoff_rule':'kickoff_utc - 4 hours',
 'eligible_history':'historical kickoff UTC strictly before cutoff, finished in retrieved archive; exclude target ID',
 'order':'most recent historical kickoffs first; ties at rank 20 require review',
 'score_field':'score.regulation, for team_id regardless of home or away',
 'history_20_or_more':'mean of last 20 eligible matches',
 'history_10_to_19':'mean of available n, with history_n in audit metadata',
 'history_under_10':'missing (null), never fill with future matches',
 'missing_score':'review; no silent substitution',
 'pagination':'continue pages until 20 eligible match histories found or all pages exhausted; verify ordering and duplicate IDs',
 'historical_status_asof_certified':False,
 'live_data_parity_certified':False,
 'training_allowed':False,
 'pilot_summary':{
  'teams':18,
  'team_match_observations':52,
  'complete20':51,
  'partial19':1,
  'score_uses':1039
 },
 'frozen_sources_sha256':SOURCES,
 'raw_cache_sha256':cache_paths,
 'g02g10_script_sha256':sha(script),
 'disclaimer':'Pilot coverage and score arithmetic validated, historical publication times not verified; not an independent model test.'
}

OUT.parent.mkdir(parents=True,exist_ok=True)

with OUT.open('x',encoding='utf-8') as f:
 json.dump(contract,f,ensure_ascii=False,sort_keys=True,indent=2)
 f.write('\n')

print('G02-G11 FREEZE OK')
print('Fichier :',OUT)
print('SHA256 :',sha(OUT))
print('0 API | aucun label | aucun entrainement')
