import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title='Human–AI Foresight', page_icon='◈', layout='wide')
APP_DIR = Path(__file__).resolve().parent

def find_file(p):
    p = Path(p)
    for x in (APP_DIR/p, APP_DIR.parent/p):
        if x.exists(): return x
    return APP_DIR/p

ARTICLES_FILE=find_file('data/processed/articles_with_topics.csv')
METRICS_FILE=find_file('data/processed/signal_metrics.csv')
TOPIC_SUMMARY_FILE=find_file('data/processed/topic_summary.csv')
EVIDENCE_FILE=find_file('data/processed/cluster_evidence_digest.csv')
AI_TRENDS_FILE=find_file('data/processed/ai_candidate_trends.csv')
REVIEWS_FOLDER=APP_DIR/'data'/'human_reviews'; REVIEWS_FOLDER.mkdir(parents=True,exist_ok=True)
required={'Articles':ARTICLES_FILE,'Signal metrics':METRICS_FILE,'Topic summary':TOPIC_SUMMARY_FILE,'Evidence digest':EVIDENCE_FILE,'AI candidate trends':AI_TRENDS_FILE}
missing=[f'{k}: {v}' for k,v in required.items() if not v.exists()]
if missing:
    st.error('Required data files could not be found.'); [st.code(x) for x in missing]; st.stop()

@st.cache_data
def load_data():
    a=pd.read_csv(ARTICLES_FILE); m=pd.read_csv(METRICS_FILE); s=pd.read_csv(TOPIC_SUMMARY_FILE); e=pd.read_csv(EVIDENCE_FILE); t=pd.read_csv(AI_TRENDS_FILE)
    for df in (a,e):
        if 'date' in df.columns: df['date']=pd.to_datetime(df['date'],errors='coerce')
    return a,m,s,e,t
articles,metrics,topic_summary,evidence,ai_trends=load_data()

def safe_text(v, default=''):
    return default if pd.isna(v) or not str(v).strip() else str(v).strip()

def get_ai_trend(topic_id):
    if 'topic_id' not in ai_trends.columns: return None
    x=ai_trends[pd.to_numeric(ai_trends.topic_id,errors='coerce')==int(topic_id)]
    return None if x.empty else x.iloc[0]

def get_topic(topic_id):
    mm=metrics[pd.to_numeric(metrics.topic_id,errors='coerce')==int(topic_id)]
    if mm.empty: st.error(f'No metrics for topic {topic_id}'); st.stop()
    ta=articles[pd.to_numeric(articles.topic_id,errors='coerce')==int(topic_id)].copy()
    if 'date' in ta: ta=ta.sort_values('date',ascending=False)
    sm=topic_summary[pd.to_numeric(topic_summary.get('Topic'),errors='coerce')==int(topic_id)] if 'Topic' in topic_summary else pd.DataFrame()
    ev=evidence[pd.to_numeric(evidence.topic_id,errors='coerce')==int(topic_id)].copy()
    if 'evidence_number' in ev: ev=ev.sort_values('evidence_number')
    return mm.iloc[0],ta,(None if sm.empty else sm.iloc[0]),ev

def participant_file(): return REVIEWS_FOLDER/f"{st.session_state.participant_id.strip()}_reviews.csv"
def evidence_file(): return REVIEWS_FOLDER/f"{st.session_state.participant_id.strip()}_evidence_reviews.csv"
def read_csv_or_empty(p):
    try: return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except Exception: return pd.DataFrame()
def upsert(path,row,keys):
    new=pd.DataFrame([row]); old=read_csv_or_empty(path)
    if not old.empty and all(k in old.columns for k in keys):
        mask=pd.Series(True,index=old.index)
        for k in keys: mask &= old[k].astype(str)==str(row[k])
        old=old[~mask]; new=pd.concat([old,new],ignore_index=True)
    new.to_csv(path,index=False,encoding='utf-8-sig')

for k,v in {'started':False,'current_index':0,'participant_id':'','participant_role':'','years_experience':0}.items():
    st.session_state.setdefault(k,v)
topic_ids=sorted(pd.to_numeric(metrics.topic_id,errors='coerce').dropna().astype(int).unique().tolist())

if not st.session_state.started:
    st.title('Human–AI Foresight')
    st.subheader('UK Sportswear · Human-in-the-Loop Review')
    st.write('The system has surfaced **candidate evidence, groupings and trend interpretations**. These are not validated trends. Your task is to decide what deserves to remain, what is noise, and how the final trend should be interpreted.')
    st.info('The purpose is not to agree with the AI. Rejection, restructuring and reframing are valuable research outcomes.')
    c1,c2=st.columns(2)
    with c1:
        pid=st.text_input('Participant ID',placeholder='e.g. P01'); role=st.text_input('Professional role',placeholder='e.g. Trend Forecaster')
    with c2: exp=st.number_input('Years of relevant experience',0,50,0)
    if st.button('Enter review →',type='primary',use_container_width=True):
        if not pid.strip(): st.error('Please enter a Participant ID.')
        else:
            st.session_state.participant_id=pid.strip(); st.session_state.participant_role=role.strip(); st.session_state.years_experience=exp; st.session_state.started=True; st.rerun()
    st.stop()

saved=read_csv_or_empty(participant_file())
if not saved.empty and saved.topic_id.nunique()>=len(topic_ids):
    st.title('HITL review complete')
    st.write('This page shows the final human-refined foresight output and the intervention data generated during review.')
    included=saved[saved.signal_decision=='Include in forecast'] if 'signal_decision' in saved else pd.DataFrame()
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Clusters reviewed',len(saved)); c2.metric('Included',len(included)); c3.metric('Modified',int(saved.get('trend_action',pd.Series()).isin(['Rename / reframe','Rewrite interpretation']).sum())); c4.metric('Rejected',int((saved.signal_decision=='Exclude from forecast').sum()))
    st.divider(); st.header('Final Human–AI Forecast')
    for n,(_,r) in enumerate(included.iterrows(),1):
        st.subheader(f"{n:02d} — {safe_text(r.get('human_signal_name'), 'Human-refined trend')}")
        st.write(safe_text(r.get('interpretation')))
        if safe_text(r.get('strategic_implication')): st.markdown('**Strategic implication**'); st.write(r.strategic_implication)
        if safe_text(r.get('missing_context')): st.markdown('**Human context added**'); st.write(r.missing_context)
    er=read_csv_or_empty(evidence_file())
    st.divider(); st.header('Intervention summary')
    if not er.empty:
        total=len(er); rejected=int((er.evidence_decision=='Reject').sum()); kept=int((er.evidence_decision=='Keep').sum())
        a,b,c=st.columns(3); a.metric('Evidence reviewed',total); b.metric('Kept',kept); c.metric('Rejected',rejected)
        st.caption(f'Evidence retention rate: {(kept/total*100):.0f}%' if total else '')
        st.download_button('Download evidence intervention data',er.to_csv(index=False).encode('utf-8'),f'{st.session_state.participant_id}_evidence_reviews.csv','text/csv')
    st.download_button('Download cluster/trend review data',saved.to_csv(index=False).encode('utf-8'),f'{st.session_state.participant_id}_reviews.csv','text/csv')
    st.stop()

idx=st.session_state.current_index; topic_id=topic_ids[idx]
metric,topic_articles,summary,topic_evidence=get_topic(topic_id); ai=get_ai_trend(topic_id)
st.title('Human-in-the-Loop Foresight Review'); st.progress((idx+1)/len(topic_ids)); st.caption(f'Candidate area {idx+1} of {len(topic_ids)}')

st.header('1 · Review the candidate evidence')
st.caption('Judge the evidence itself before judging the AI trend. Fast retrieval does not imply foresight relevance.')
for j,(_,row) in enumerate(topic_evidence.iterrows(),1):
    evid_id=safe_text(row.get('evidence_number'),j); title=safe_text(row.get('title'),'Untitled'); source=safe_text(row.get('source'),'Unknown source'); date=row.get('date'); d=date.strftime('%d %b %Y') if pd.notna(date) and hasattr(date,'strftime') else 'Unknown date'
    with st.expander(f'{j}. {title}',expanded=(j==1)):
        st.caption(f'{source} · {d}'); txt=safe_text(row.get('available_text')); st.write(txt[:900] if txt else 'No extract available.')
        url=safe_text(row.get('url')); 
        if url: st.markdown(f'[Open original source]({url})')
        decision=st.radio('Evidence decision',['Keep','Reject','Unsure'],index=None,horizontal=True,key=f'ed_{topic_id}_{evid_id}')
        reason=''
        if decision=='Reject': reason=st.selectbox('Why reject it?',['Advertorial / commercial content','Weak evidence','Not a foresight signal','Irrelevant to the question','Duplicate / repetitive','Poor or unclear source','Other'],index=None,key=f'er_{topic_id}_{evid_id}')
        note=st.text_input('Optional note',key=f'en_{topic_id}_{evid_id}')
        if decision:
            upsert(evidence_file(),{'participant_id':st.session_state.participant_id,'topic_id':topic_id,'evidence_id':evid_id,'title':title,'source':source,'evidence_decision':decision,'rejection_reason':reason or '','human_note':note,'timestamp':datetime.now().isoformat()},['participant_id','topic_id','evidence_id'])

st.divider(); st.header('2 · Judge the AI grouping')
st.write(f"The computational pipeline grouped **{int(metric.get('total_articles',0))} items from {int(metric.get('unique_sources',0))} sources** into this candidate area.")
grouping_action=st.radio('Does this grouping make sense?',['Keep grouping','Reframe grouping','Merge with another candidate area','Reject grouping'],index=None,key=f'grp_{topic_id}')
grouping_reason=st.text_area('Why?',placeholder='What relationship is meaningful, weak or missing?',key=f'grpr_{topic_id}')
merge_target=''
if grouping_action=='Merge with another candidate area': merge_target=st.selectbox('Merge with', [f'Candidate {i+1} (topic {t})' for i,t in enumerate(topic_ids) if t!=topic_id],key=f'merge_{topic_id}')

st.divider(); st.header('3 · Judge the AI interpretation')
if ai is None: st.warning('No AI candidate interpretation was generated for this topic.'); ai_name='No AI trend'; ai_hyp=''
else:
    ai_name=safe_text(ai.get('ai_trend_name'),f'Candidate {idx+1}'); ai_hyp=safe_text(ai.get('ai_hypothesis'))
    st.subheader(ai_name); st.markdown('**AI hypothesis**'); st.write(ai_hyp or 'No hypothesis supplied.')
    rationale=safe_text(ai.get('ai_rationale')); limitations=safe_text(ai.get('limitations'))
    if rationale:
        with st.expander('Why the AI proposed this'): st.write(rationale)
    if limitations:
        with st.expander('AI limitations / counter-evidence'): st.write(limitations)
trend_action=st.radio('What should happen to this AI interpretation?',['Accept unchanged','Rename / reframe','Rewrite interpretation','Reject AI interpretation'],index=None,key=f'ta_{topic_id}')

st.divider(); st.header('4 · Professional interpretation')
classification=st.radio('How would you classify the underlying pattern?',['Emerging signal','Already established','Relevant context, but not a signal','Noise / unrelated','Need more evidence'],index=None,key=f'cl_{topic_id}')
signal_decision=st.radio('Final forecast decision',['Include in forecast','Exclude from forecast','Need more evidence'],index=None,horizontal=True,key=f'dec_{topic_id}')
human_name=interpretation=missing_context=strategic=future=rejection=''
if signal_decision=='Include in forecast':
    human_name=st.text_input('Final human-refined trend name',value='' if trend_action!='Accept unchanged' else ai_name,key=f'name_{topic_id}')
    interpretation=st.text_area('Final interpretation — what change does this evidence represent?',key=f'int_{topic_id}')
    missing_context=st.text_area('What professional/cultural context did the computational analysis miss?',key=f'ctx_{topic_id}')
    strategic=st.text_area('Why does this matter strategically for UK sportswear?',key=f'strat_{topic_id}')
    future=st.text_area('How could this develop over the next 2–3 years?',key=f'fut_{topic_id}')
elif signal_decision=='Exclude from forecast':
    rejection=st.selectbox('Why exclude the overall candidate?',['Evidence too weak / noisy','Articles do not belong together','Already established','Not relevant enough','Not strategically meaningful','AI interpretation unsupported','Other'],index=None,key=f'rej_{topic_id}')
confidence=st.slider('Confidence in your judgement',1,5,3,key=f'conf_{topic_id}')

st.divider(); back,nextc=st.columns([1,3])
with back:
    if idx>0 and st.button('← Previous'): st.session_state.current_index-=1; st.rerun()
with nextc:
    if st.button('Save & Next →',type='primary',use_container_width=True):
        errors=[]
        er=read_csv_or_empty(evidence_file()); reviewed=er[pd.to_numeric(er.get('topic_id'),errors='coerce')==topic_id] if not er.empty else pd.DataFrame()
        if len(reviewed)<len(topic_evidence): errors.append('Please make a Keep / Reject / Unsure decision for every representative evidence item.')
        if grouping_action is None: errors.append('Please judge the AI grouping.')
        if trend_action is None: errors.append('Please judge the AI interpretation.')
        if classification is None or signal_decision is None: errors.append('Please complete the professional classification and final forecast decision.')
        if signal_decision=='Include in forecast' and not human_name.strip(): errors.append('Please provide a final trend name.')
        if signal_decision=='Exclude from forecast' and not rejection: errors.append('Please provide an exclusion reason.')
        if errors:
            for e in errors: st.error(e)
        else:
            row={'participant_id':st.session_state.participant_id,'participant_role':st.session_state.participant_role,'years_experience':st.session_state.years_experience,'topic_id':topic_id,'ai_trend_name':ai_name,'ai_hypothesis':ai_hyp,'grouping_action':grouping_action,'grouping_reason':grouping_reason,'merge_target':merge_target,'trend_action':trend_action,'classification':classification,'signal_decision':signal_decision,'human_signal_name':human_name,'interpretation':interpretation,'missing_context':missing_context,'strategic_implication':strategic,'future_development':future,'rejection_reason':rejection,'confidence':confidence,'machine_total_articles':metric.get('total_articles'),'machine_unique_sources':metric.get('unique_sources'),'machine_growth_percent':metric.get('growth_percent'),'timestamp':datetime.now().isoformat()}
            upsert(participant_file(),row,['participant_id','topic_id'])
            if idx<len(topic_ids)-1: st.session_state.current_index+=1
            st.rerun()
