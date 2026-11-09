#!/usr/bin/env python3
"""Verify this portable corpus and reproduce descriptive annotation aggregates (stdlib only)."""
from pathlib import Path
import argparse,collections,gzip,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parent
SHA=lambda b:hashlib.sha256(b).hexdigest()
def read_bytes(path):return (ROOT/path).read_bytes()
def obj(path):
 b=read_bytes(path);return json.loads(gzip.decompress(b) if str(path).endswith('.gz') else b)
def rows(path):
 b=read_bytes(path);b=gzip.decompress(b) if str(path).endswith('.gz') else b
 return [json.loads(s) for s in b.decode('utf-8').splitlines() if s.strip()]
def norm(s):return ' '.join(s.split())
def require(ok,msg):
 if not ok:raise AssertionError(msg)
def counts(rs):
 return {'n':len(rs),'unique_trajectories':len({r['trajectory'] for r in rs}),'verdict_codes':dict(collections.Counter(r['verdict_code'] for r in rs)),'reflection_codes':dict(collections.Counter(r['reflection_code'] for r in rs)),'task_confusion':dict(collections.Counter(r['task_confusion'] for r in rs)),'invented':dict(collections.Counter(r['invented'] for r in rs)),'reward_misread':dict(collections.Counter(r['reward_misread'] for r in rs)),'confidence':dict(collections.Counter(r['confidence'] for r in rs)),'accept_with_rejection_rationale_V1V2':sum(r['verdict']=='accept' and r['verdict_code'] in ['V1','V2'] for r in rs),'n_with_reflection':sum(r['reflection_code']!='-' for r in rs)}
def aggregate(coded):
 groups={'student_designs':lambda x:x['src']=='student','student_original':lambda x:x['src']=='original','teacher':lambda x:x['src']=='teacher','astra':lambda x:x['session']=='collusion_astra','opus':lambda x:x['session']=='collusion_opus'}
 s={k:{'constrained':counts([r for r in coded if pred(r) and r['constrained']]),'other_rounds':counts([r for r in coded if pred(r) and not r['constrained']])} for k,pred in groups.items()}
 s['all']={'constrained':counts([r for r in coded if r['constrained']]),'other_rounds':counts([r for r in coded if not r['constrained']])}
 s['per_test']={a+'/'+b:counts([r for r in coded if r['session']==a and r['test']==b and r['constrained']]) for a,b in sorted({(r['session'],r['test']) for r in coded if r['src']=='student'})}
 return s

def check_private(v,p=''):
 if isinstance(v,dict):
  for k,x in v.items():
   q=p+'/'+str(k)
   if str(k).lower() in {'api_key','apikey','api-key','authorization','access_token','refresh_token','client_secret','secret_access_key','aws_secret_access_key','bearer_token','auth_token'} and x:
    require(x=='<REDACTED_CREDENTIAL>','Unredacted credential metadata at '+q)
   check_private(x,q)
 elif isinstance(v,list):
  for x in v:check_private(x,p+'/*')
 elif isinstance(v,str):
  for marker in ['/Users/','/private/tmp/','-Users-','/workspace/agent-collusion','/opt/anaconda3']:
   require(marker not in v,'Residual private/infrastructure path at '+p)

def verify(write_summary=None):
 files=obj('SHA256SUMS.json')['files']
 actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json' and '__pycache__' not in p.parts}
 require(actual==set(files),'Checksum inventory differs from actual bundle files')
 for name,spec in files.items():
  b=read_bytes(name);require(SHA(b)==spec['sha256'],'Checksum mismatch: '+name);require(len(b)==spec['bytes'],'Byte length mismatch: '+name)
  if name.endswith('.gz'):require(b[4:8]==b'\x00\x00\x00\x00','Nondeterministic gzip mtime: '+name)
 corpus=rows('corpus/agent_episodes.jsonl.gz');coded=rows('annotations/calibrated.jsonl.gz')
 by={r['record_id']:r for r in corpus};ab={r['record_id']:r for r in coded}
 require(len(corpus)==len(by)==736,'Corpus count/unique key mismatch');require(len(coded)==len(ab)==736 and set(by)==set(ab),'Annotation coverage mismatch')
 require(sum(bool(r['constrained']) for r in corpus)==640,'Constrained scope mismatch');require(sum(not r['constrained'] for r in corpus)==96,'Other-round scope mismatch')
 require(len({r['trajectory'] for r in corpus})==49,'Trajectory count mismatch')
 required={'record_id','verdict_code','invented','reflection_code','task_confusion','reward_misread','confidence','verdict_quote','reflection_quote','note','reviewed_by'}
 for rid,r in by.items():
  a=ab[rid];require(required<=set(a),'Missing annotation fields: '+rid)
  for k,valid in [('verdict_code',{'V1','V2','V3','V4','V5','V6'}),('reflection_code',{'R1','R2','R3','R4','-'}),('confidence',{'high','medium','low'}),('invented',{'yes','no','unclear'}),('task_confusion',{'yes','no','unclear'}),('reward_misread',{'yes','no','unclear'})]:require(a[k] in valid,'Invalid '+k+': '+rid)
  for q,s in [('verdict_quote','reasoning'),('reflection_quote','reflection')]:
   require(len(a[q].split())<=25 and norm(a[q]) in norm(r[s]),'Quote provenance/length mismatch: '+rid+' '+q)
  require(bool(a['verdict_quote']),'Missing verdict quote: '+rid)
  require(not r['reflection'] or bool(a['reflection_quote']),'Missing reflection quote: '+rid)
  require(r['reflection'] or a['reflection_code']=='-','Absent reflection coded: '+rid)
  require(a['verdict_code']!='V3' or a['task_confusion']=='yes','V3 flag mismatch: '+rid)
  require(len(a['note'].split())<=60,'Long note: '+rid)
  for k in ['src','session','test','trajectory','ep','agent','constrained','verdict_policy','throttled','verdict','reward','reward_scheme','peer_expected','source_sha256']:require(a[k]==r[k],'Joined annotation metadata mismatch: '+rid+' '+k)
  require(a['received_messages']==len(r['channel']) and a['failed_sends']==len(r['bounced_sends']),'Channel count mismatch: '+rid)
  require(a['source_file']==r['source_path'],'Source reference mismatch: '+rid)
 reviewer=[]
 for path in sorted((ROOT/'annotations/reviewer_codes').glob('*.jsonl.gz')):reviewer.extend(rows(path.relative_to(ROOT)))
 rb={r['record_id']:r for r in reviewer};require(len(reviewer)==len(rb)==736 and set(rb)==set(by),'Reviewer annotation coverage mismatch')
 for rid,a in rb.items():
  for k in required:require(a[k]==ab[rid][k],'Reviewer/calibrated mismatch: '+rid+' '+k)
 source_manifest=obj('provenance/source_manifest.json');sources=source_manifest['complete_collusion_runs'];require(len(sources)==49,'Full-source run count mismatch')
 covered=set();private_logs=0
 for s in sources:
  b=read_bytes(s['portable_path']);raw=gzip.decompress(b)
  require(SHA(b)==s['sanitized_gzip_bytes_sha256'] and SHA(raw)==s['sanitized_json_bytes_sha256'],'Source sanitized hash mismatch: '+s['source_id'])
  run=json.loads(raw);check_private(run)
  for endpoint in (run.get('run_config') or {}).get('endpoints',{}).values():
   if isinstance(endpoint,dict):
    for k in ['api_base','base_url']:
     if endpoint.get(k):require(endpoint[k]=='<REDACTED_ENDPOINT>','Unredacted inference endpoint')
  for rid in s['record_ids']:
   require(rid in by and rid not in covered,'Source duplicate/unknown record: '+rid);covered.add(rid);r=by[rid]
   require(r['source_path']==s['portable_path'] and r['source_sha256']==s['original_bytes_sha256'],'Source link/hash mismatch: '+rid)
   e=run['results'][r['source_episode_index']];agent=e['agents'][r['agent']]
   require(((agent.get('verdict_submission') or {}).get('reasoning') or '')==r['reasoning'],'Full-run reasoning mismatch: '+rid)
   require((agent.get('reflection') or '')==r['reflection'],'Full-run reflection mismatch: '+rid)
   require(agent.get('outcome_feedback')==r['outcome_feedback'],'Full-run feedback mismatch: '+rid)
   require(agent.get('verdict')==r['verdict'],'Full-run verdict mismatch: '+rid)
   require([x for x in (e.get('channel_transcript') or []) if x.get('receiver')==r['agent']]==r['channel'],'Delivered channel mismatch: '+rid)
   require((e.get('channel_transcript') or [])==r['all_channel'],'All-channel mismatch: '+rid)
   require([x for x in (e.get('events') or []) if x.get('actor')==r['agent'] and x.get('tool')=='send_message' and x.get('success') is False]==r['bounced_sends'],'Failed-send mismatch: '+rid)
   private_logs+=bool(agent.get('private_raw_log'))
 require(covered==set(by),'Full-source coverage incomplete')
 selected=obj('casebank/selected_cases.json.gz')
 require((len(selected['collusion_records']),len(selected['acb_turns']),len(selected['designer_notes']))==(64,4,8),'Casebank count mismatch')
 require(len({x['record_id'] for x in selected['collusion_records']})==64,'Duplicate selected case')
 for x in selected['collusion_records']:
  for k,v in x.items():require(v==by[x['record_id']][k],'Casebank/corpus mismatch: '+x['record_id']+' '+k)
 for key,path in [('collusion_records','casebank/collusion_records.jsonl.gz'),('acb_turns','casebank/acb_turns.jsonl.gz'),('designer_notes','casebank/designer_notes.jsonl.gz')]:require(selected[key]==rows(path),'Casebank split-file mismatch: '+key)
 for x in selected['designer_notes']:
  require(read_bytes(x['source_path']).decode()==x['text'],'Designer-note mismatch: '+x['workspace'])
 for x in selected['acb_turns']:
  require(x['trace_index_zero_based']>=0 and isinstance(x['turn']['content'],str),'Invalid selected ACB turn')
  require(x['assigned_role']['agent_id']==x['turn']['agent_id'],'ACB assigned-agent mismatch')
 for path in sorted(ROOT.rglob('*')):
  if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ['.json','.gz']:continue
  if path.name=='redactions.jsonl.gz':vals=rows(path.relative_to(ROOT))
  elif path.name.endswith('.jsonl.gz'):vals=rows(path.relative_to(ROOT))
  else:vals=obj(path.relative_to(ROOT))
  check_private(vals)
 supplement=obj('provenance/designer_supplement_manifest.json')
 require(supplement['generated_astra_event_extract_files']==4 and supplement['generated_astra_event_records']==435,'Designer extract scope mismatch')
 require(supplement['workspace_root_json_files']==42 and supplement['test_result_summary_files']==41 and supplement['session_ledger_files']==8 and supplement['session_ledger_records']==100,'Designer workspace/ledger scope mismatch')
 require(supplement['full_opus_designer_transcripts_included'] is False,'Unsupported full Opus transcript claim')
 for item in supplement['files']:
  b=read_bytes(item['published_path']);raw=gzip.decompress(b)
  require(SHA(b)==item['sanitized_gzip_bytes_sha256'] and SHA(raw)==item['sanitized_json_bytes_sha256'],'Designer supplement hash mismatch: '+item['published_path'])
  data=rows(item['published_path']) if item['published_path'].endswith('.jsonl.gz') else json.loads(raw)
  if 'records' in item:require(len(data)==item['records'],'Designer record count mismatch')
  if item['kind']=='generated_natural_language_event_extract':
   require(all(x['type']=='agent_message' and isinstance(x['source_line'],int) and isinstance(x['text'],str) for x in data),'Invalid generated designer event extract')
  for stamp in item['recorded_timestamps']:
   value=data
   for token in stamp['json_pointer'].strip('/').split('/'):
    value=value[int(token)] if isinstance(value,list) else value[token]
   require(value==stamp['value'],'Recorded timestamp changed: '+item['published_path'])
 summary=aggregate(coded);require(summary==obj('annotations/classification_summary.json'),'Recomputed descriptive aggregate differs')
 validation=obj('annotations/coding_validation.json');require(validation['n_input']==validation['n_coded']==validation['n_unique_coded']==736 and not validation['missing'] and not validation['errors'],'Saved coding validation failed')
 if write_summary:Path(write_summary).write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 result={'status':'pass','corpus_records':736,'constrained_records':640,'other_round_records':96,'full_source_runs':49,'source_linked_records':len(covered),'records_with_private_task_raw_log':private_logs,'casebank':{'collusion_records':64,'acb_turns':4,'designer_notes':8},'annotation_quotes_and_labels':'pass','source_rationale_feedback_action_channels':'pass','recomputed_aggregate':'pass','checksum_inventory':'pass','explicit_infrastructure_redaction_checks':'pass','designer_supplement':{'astra_agent_messages':435,'saved_specs_probes_configs':42,'test_result_summaries':41,'session_ledgers':8,'ledger_events':100},'annotation_status':'Agent-generated calibrated annotations; not human gold labels.','scope_limit':'Four selected ACB turns are included; their complete ACB source files belong to the separate ACB release, not this bundle.'}
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--write-summary',help='Optional destination for the recomputed descriptive aggregate JSON');args=ap.parse_args()
 try:print(json.dumps(verify(args.write_summary),ensure_ascii=False,sort_keys=True,indent=2))
 except (AssertionError,KeyError,ValueError,OSError) as e:print('Verification failed: '+str(e),file=sys.stderr);sys.exit(1)
