#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

B = Path('outputs/v4_research')
PILOT = B/'v42_g02g6_prematch_pilot'
TARGETS = B/'v42_g02g5_prematch_plan/G02G5_TARGET_OBSERVATIONS_FROZEN.csv'
MANIFEST = B/'v41_mass350/manifest.csv'
PLAN = PILOT/'G02G6_PILOT6_TEAMS_FROZEN.csv'
PAGE2 = PILOT/'raw_history_pages/tm_279303_page_2.json'

EXPECTED = {
    TARGETS:'4180cd7b325e299632658fe6cd63cb2b391dc93405e2bb90799e59adb697ccf2',
    MANIFEST:'61417a2ef02884af6a75fd2a8926961d9b59208fb0d4ce352433ecfe4eeada7e',
    PLAN:'a136f915f5cc87106d085d162660eb699eb2282f904929ef9cc21067471f24a0',
    PAGE2:'6a65a6dc905e7ab9cd51781dc2d933d20b0828771a56f91ab8c3e872078424f9',
}

TEAM_HASHES = {
    'tm_37445':'c3900ea631f71c7f65405137c0b6f67d1f503f6774a6f6e29f5959671c8d7daf',
    'tm_087341':'1693cd6bf92bf704c2317f90ac40079d217ed72cfc3bc95b0f50803dd9599c87',
    'tm_279303':'40561207f351e74bd6e3dff22f77df04c86d5983c1ffc7bd7d68c3e5277e2c12',
    'tm_46516':'29b25d63a5bc29cc510abb843aa97143d923855fc102ceaa909f18ee6d48d15d',
    'tm_19704':'7f04081190aca47d1a817fa0ad11cd83602433b2cf6dee08d6b423b216289b73',
    'tm_6047':'adef09c91a4f33bf8c846414dc7a5b6591610c2f39100e8f8d841efe6e595c67',
}

def stop(s):
    raise SystemExit('STOP : '+s)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def utc(v):
    d = datetime.fromisoformat(str(v).replace('Z','+00:00'))
    if d.tzinfo is None:
        stop('datetime sans fuseau '+str(v))
    return d.astimezone(timezone.utc)

def integer_score(v):
    if isinstance(v,bool) or v is None:
        return None
    try:
        x=float(v)
        if math.isfinite(x) and x>=0 and x.is_integer():
            return int(x)
    except (TypeError,ValueError):
        pass
    return None

def goals(m, team):
    home = m.get('home_team') or {}
    away = m.get('away_team') or {}

    if not isinstance(home,dict) or not isinstance(away,dict):
        return None,'BAD_TEAMS',False

    hid,aid=home.get('id'),away.get('id')

    if (hid==team)+(aid==team)!=1:
        return None,'WRONG_TEAM',False

    side='home' if hid==team else 'away'
    s=m.get('score')

    if not isinstance(s,dict):
        return None,'MISSING_SCORE',False

    reg=s.get('regulation')
    reg_val=integer_score(reg.get(side)) if isinstance(reg,dict) else None
    top_val=integer_score(s.get(side))

    if reg_val is not None:
        return reg_val,'REGULATION',top_val is not None and top_val!=reg_val

    if top_val is not None:
        return top_val,'TOP_LEVEL_FALLBACK',False

    return None,'MISSING_GOALS',False

for p,expected in EXPECTED.items():
    if not p.is_file() or sha(p)!=expected:
        stop('SHA source absente ou modifiee : '+str(p))

manifest={r['match_id']:r for r in csvrows(MANIFEST)}
if len(manifest)!=12415:
    stop('manifest invalide')

if {r['team_id'] for r in csvrows(PLAN)}!=set(TEAM_HASHES):
    stop('selection pilote invalide')

obs=[r for r in csvrows(TARGETS) if r['team_id'] in TEAM_HASHES]

if len(obs)!=22 or len({(r['match_id'],r['side']) for r in obs})!=22:
    stop('observations invalides')

hist={}
for team,expected in TEAM_HASHES.items():
    p=PILOT/'raw_history'/(team+'.json')
    if not p.is_file() or sha(p)!=expected:
        stop('SHA cache incorrect : '+team)

    wrapper=json.loads(p.read_text(encoding='utf-8'))
    if wrapper.get('team_id')!=team:
        stop('cache equipe incorrect : '+team)

    response=wrapper.get('response')
    if not isinstance(response,dict) or int(response.get('meta',{}).get('page',-1))!=1:
        stop('page 1 invalide : '+team)

    data=response.get('data')
    if not isinstance(data,list) or len(data)!=100:
        stop('100 historiques attendus : '+team)

    entries=[]
    for m in data:
        if m.get('status')!='finished':
            stop('historique non termine : '+team)
        entries.append((utc(m['utc_date']),m['id'],m))

    if len(set(i for _,i,_ in entries))!=100:
        stop('doublons page 1 : '+team)

    if any(entries[i][0]<entries[i+1][0] for i in range(99)):
        stop('ordre inattendu page 1 : '+team)

    hist[team]=entries

extra_data=json.loads(PAGE2.read_text(encoding='utf-8'))
if extra_data.get('team_id')!='tm_279303' or len(extra_data['response']['data'])!=1:
    stop('page 2 invalide')

extra=extra_data['response']['data'][0]
if extra['id'] in {x[1] for x in hist['tm_279303']}:
    stop('doublon entre pages')

print('='*76)
print('G02-G6F | AUDIT DES SCORES SUR LES 20 DERNIERS MATCHS')
print('='*76)

counts=Counter()
examples=[]

try:
    import rebuild_clean_prematch_v2 as pre
    lab=pre.load_lab()
    historical_check=True
except Exception as exc:
    print('Comparaison constructeur historique indisponible :',
          type(exc).__name__,str(exc))
    historical_check=False

for r in sorted(obs,key=lambda z:(z['team_id'],z['safe_cutoff_utc'],z['match_id'])):
    team,mid,side=r['team_id'],r['match_id'],r['side']

    if mid not in manifest or manifest[mid][side+'_id']!=team:
        stop('jointure incorrecte : '+mid)

    cutoff=utc(r['safe_cutoff_utc'])
    kickoff=utc(manifest[mid]['kickoff_utc'])

    if kickoff-cutoff!=timedelta(hours=4):
        stop('marge non conforme : '+mid)

    prior=sorted(
        (e for e in hist[team] if e[0]<cutoff and e[1]!=mid),
        key=lambda x:(x[0],x[1]),
        reverse=True
    )
    selected=prior[:20]

    if len(selected)!=20:
        stop('moins de 20 historiques : '+mid)

    if team=='tm_279303':
        e=(utc(extra['utc_date']),extra['id'],extra)
        full=sorted(
            prior+([e] if e[0]<cutoff and e[1]!=mid else []),
            key=lambda x:(x[0],x[1]),
            reverse=True
        )
        if [x[1] for x in full[:20]]!=[x[1] for x in selected]:
            stop('top20 modifie page 2 : '+mid)
        counts['pagination_top20_unchanged']+=1

    scored=[]
    sources=Counter()

    for d,history_mid,m in selected:
        v,origin,disagree=goals(m,team)
        sources[origin]+=1
        counts['score_sources_'+origin]+=1

        if disagree:
            counts['top_reg_differences']+=1
            if len(examples)<10:
                examples.append(
                    (team,mid,history_mid,'TOP_VS_REG',m.get('score'))
                )

        if v is None:
            counts['unusable_histories']+=1
            if len(examples)<10:
                examples.append(
                    (team,mid,history_mid,origin,m.get('score'))
                )
        else:
            scored.append(v)

    counts['observations']+=1
    counts['selected_match_observations']+=len(selected)
    counts['observations_complete_scores']+=len(scored)==20

    mean=sum(scored)/20 if len(scored)==20 else None
    historical=None

    if historical_check:
        try:
            historical=pre.build_team_summary(
                lab,
                [e[2] for e in selected],
                team,
                manifest[mid][side+'_team']
            ).get('avg_goals_scored')

            if mean is not None and historical is not None and abs(mean-float(historical))>1e-9:
                counts['different_from_historical_builder']+=1
                if len(examples)<10:
                    examples.append(
                        (team,mid,'MEAN_DISAGREEMENT',mean,historical)
                    )

            if historical is None:
                counts['historical_builder_missing']+=1

        except Exception as exc:
            counts['historical_builder_error']+=1
            if len(examples)<10:
                examples.append(
                    (team,mid,'BUILDER_ERROR',type(exc).__name__,str(exc)[:100])
                )

    lead=(cutoff-selected[0][0]).total_seconds()/3600
    if lead<0:
        stop('future historique : '+mid)

    counts['min_lead_hours']=min(
        counts.get('min_lead_hours',float('inf')),lead
    )

    print(
        team,mid,side,
        '| scores =',str(len(scored))+'/20',
        '| moyenne =',round(mean,6) if mean is not None else None,
        '| ancienne methode meme20 =',
        round(float(historical),6) if historical is not None else None,
        '| origins =',dict(sources)
    )

print('\n=== BILAN ===')
for k,v in sorted(counts.items()):
    print(k,':',v)

print('Exemples de cas a examiner :',examples)

print('\nLIMITES : finished aujourd hui ne certifie pas finished avant le cutoff.')
print('Les archives ne contiennent pas d heure de publication des resultats.')
print('Scores/revisions de la source et representativite des 3000 observations non certifies.')
print('AUCUNE API. AUCUNE ECRITURE. AUCUN ENTRAINEMENT.')
print('G02-G6F TERMINE')
