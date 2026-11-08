import csv, copy, hashlib, json, math
from pathlib import Path
from collections import Counter
import build_historical_ab_research_v3 as hist

B = Path('outputs/v4_research')
REG = B/'v42_g02f2_quality_pre87/G02F2_PRE87_QUALITY_REGISTRY.csv'
F6 = B/'v42_g02f6_live8_adapter/V42_LIVE8_ADAPTED_RESEARCH.csv'
MAN = B/'v41_mass350/manifest.csv'
EXPECTED = {
    REG: '4cfe6146db28a4f05837bd4b66673eb146099bf7363197e5096a51de3cdd1b17',
    F6: '2656ff84f9096b7ec8a1f7b8517b1c9c70d8f339f210a2d20ad2e4375c91b146',
    MAN: '61417a2ef02884af6a75fd2a8926961d9b59208fb0d4ce352433ecfe4eeada7e',
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with p.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    ids=[r['match_id'] for r in rows]
    if not all(ids) or len(ids)!=len(set(ids)): raise SystemExit('STOP duplicate IDs: '+str(p))
    return {r['match_id']:r for r in rows}
def equivalent(a,b):
    if a is None or str(a).strip()=='' : return b is None or str(b).strip()==''
    if b is None or str(b).strip()=='': return False
    try:
        x,y=float(a),float(b)
        return math.isfinite(x) and math.isfinite(y) and abs(x-y)<1e-6
    except (ValueError,TypeError): return False

def future(e):
    if not isinstance(e,dict): return False
    p=str(e.get('period','')).lower()
    if p in hist.EXTRA_TIME or p=='penalties': return True
    if p!='second_half': return False
    try: return hist.event_minute(e)>=87
    except (ValueError,TypeError): return False

def recent(events):
    sh=so=co=0; last=None
    for e in events:
        if not isinstance(e,dict): continue
        typ=str(e.get('type') or '').lower()
        if typ not in hist.SHOTS and typ!='corner_kick': continue
        if str(e.get('period') or '').lower() not in ('first_half','second_half'): continue
        try: minute=hist.event_minute(e)
        except (ValueError, TypeError): continue
        if minute>=87: continue
        if typ in hist.SOT: last = minute if last is None else max(last,minute)
        if e.get('period')=='second_half' and 80<=minute<87:
            sh+=int(typ in hist.SHOTS)
            so+=int(typ in hist.SOT)
            co+=int(typ=='corner_kick')
    return {'shots_80_87':sh,'sot_80_87':so,'corners_80_87':co,
            'minutes_since_last_shot_on_target':None if last is None else round(87-last,6)}

for p,h in EXPECTED.items():
    if not p.is_file() or sha(p)!=h: raise SystemExit('STOP hash changed: '+str(p))
registry,features,manifest=load(REG),load(F6),load(MAN)
if len(registry)!=1500 or len(features)!=1500 or len(manifest)!=12415 or set(registry)!=set(features):
    raise SystemExit('STOP mismatch of counts/IDs')
base=['score_87_home','score_87_away','home_shots_on_target_87','away_shots_on_target_87']
counts=Counter(); examples=[]
for mid in sorted(registry):
    r=registry[mid]; path=Path(r['source_path'])
    if path.stem!=mid or sha(path)!=r['source_sha256']:
        raise SystemExit('STOP timeline changed: '+mid)
    obj=json.loads(path.read_text(encoding='utf-8'))
    if obj.get('match_id')!=mid: raise SystemExit('STOP ID: '+mid)
    data=obj['timeline_response']['data']; events=data['events']
    original=copy.deepcopy(data)
    masked=copy.deepcopy(data)
    masked['events']=[e for e in events if not future(e)]
    counts['future_events_removed']+=len(events)-len(masked['events'])
    counts['matches_with_future_events']+=int(len(events)!=len(masked['events']))
    for e in events:
        if isinstance(e,dict) and e.get('period')=='second_half':
            try: hist.event_minute(e)
            except (ValueError,TypeError): counts['events_with_undated_second_half']+=1
    info={k:'' if v is None else str(v) for k,v in manifest[mid].items()}
    try:
        _, b0, q0=hist.build_match(mid,info,original)
        _, b1, q1=hist.build_match(mid,info,masked)
        if not isinstance(b0,dict) or not isinstance(b1,dict): raise ValueError('missing builder dictionary')
        if q0.get('status')!=q1.get('status'):
            counts['builder_status_changed_after_mask']+=1
            if len(examples)<12: examples.append((mid,'STATUS',q0.get('status'),q1.get('status')))
        for feature in base:
            if not equivalent(b0.get(feature),b1.get(feature)) or not equivalent(b0.get(feature), features[mid].get(feature)):
                counts['feature_failures']+=1
                if len(examples)<12: examples.append((mid,feature,b0.get(feature),b1.get(feature)))
        d0,d1=recent(events),recent(masked['events'])
        for feature in d0:
            if not equivalent(d0[feature],d1[feature]) or not equivalent(d0[feature],features[mid].get(feature)):
                counts['feature_failures']+=1
                if len(examples)<12: examples.append((mid,feature,d0[feature],d1[feature]))
    except Exception as exc:
        counts['exceptions']+=1
        if len(examples)<12: examples.append((mid,'EXCEPTION',type(exc).__name__,str(exc)[:90]))
    counts['matches_tested']+=1
print('='*72)
print('G02-S1 | TEST DE FUITE PAR SUPPRESSION DES EVENEMENTS FUTURS')
print('='*72)
for key in ('matches_tested','matches_with_future_events','future_events_removed',
            'feature_failures','builder_status_changed_after_mask',
            'events_with_undated_second_half','exceptions'):
    print(f'{key}: {counts[key]}')
print('Exemples:',examples)
print('NOTE: test de non-regression, PAS certification totale du live ni des labels.')
print('READ ONLY; aucun appel API; aucune ecriture.')

