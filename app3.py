import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title='Human–AI Foresight · Gen X', page_icon='◈', layout='wide')

# Simple visual style
st.markdown('''
<style>
.block-container{max-width:1250px;padding-top:2rem;padding-bottom:4rem}
.hero{padding:28px 30px;border-radius:22px;background:linear-gradient(120deg,#17131f,#25172f 55%,#151b25);border:1px solid rgba(255,255,255,.10);margin-bottom:24px}
.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.76rem;opacity:.68;margin-bottom:8px}
.ai-box{padding:18px 20px;border-radius:16px;background:linear-gradient(135deg,rgba(132,79,255,.10),rgba(45,189,255,.05));border:1px solid rgba(150,120,255,.25);margin-bottom:14px}
.context-box{padding:17px 18px;border-radius:16px;background:rgba(210,196,165,.07);border:1px solid rgba(210,196,165,.17);margin-bottom:16px}
.swatch-row{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 14px 0}
.swatch{width:82px;height:82px;border-radius:14px;border:1px solid rgba(255,255,255,.18);display:flex;align-items:flex-end;padding:8px;font-size:.68rem;color:white;text-shadow:0 1px 4px rgba(0,0,0,.7);box-sizing:border-box}
.texture{min-height:90px;border-radius:14px;padding:14px;border:1px solid rgba(255,255,255,.12);background:repeating-linear-gradient(135deg,rgba(255,255,255,.05) 0px,rgba(255,255,255,.05) 4px,rgba(255,255,255,.015) 4px,rgba(255,255,255,.015) 10px)}
.tag{display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(217,194,130,.12);border:1px solid rgba(217,194,130,.24);font-size:.8rem;margin-right:6px}
.small{opacity:.68;font-size:.82rem}
</style>
''', unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent

def find_file(relative_path):
    for p in [APP_DIR / relative_path, APP_DIR.parent / relative_path]:
        if p.exists():
            return p
    return APP_DIR / relative_path

ARTICLES_FILE = find_file(Path('data/processed/articles_with_topics.csv'))
METRICS_FILE = find_file(Path('data/processed/signal_metrics.csv'))
TOPIC_SUMMARY_FILE = find_file(Path('data/processed/topic_summary.csv'))
EVIDENCE_FILE = find_file(Path('data/processed/cluster_evidence_digest.csv'))
AI_FILE = find_file(Path('data/ai_outputs/ai_signal_assessments.csv'))
REVIEWS_FOLDER = APP_DIR / 'data' / 'human_reviews_stage2'
REVIEWS_FOLDER.mkdir(parents=True, exist_ok=True)

required = {'Articles': ARTICLES_FILE, 'Signal metrics': METRICS_FILE, 'Topic summary': TOPIC_SUMMARY_FILE, 'Evidence digest': EVIDENCE_FILE}
missing = [f'{k}: {v}' for k,v in required.items() if not v.exists()]
if missing:
    st.error('Some required data files could not be found.')
    for item in missing:
        st.code(item)
    st.stop()

@st.cache_data
def load_data():
    articles = pd.read_csv(ARTICLES_FILE)
    metrics = pd.read_csv(METRICS_FILE)
    topic_summary = pd.read_csv(TOPIC_SUMMARY_FILE)
    evidence = pd.read_csv(EVIDENCE_FILE)
    ai = pd.read_csv(AI_FILE) if AI_FILE.exists() else pd.DataFrame()
    articles['date'] = pd.to_datetime(articles['date'], errors='coerce')
    evidence['date'] = pd.to_datetime(evidence['date'], errors='coerce')
    return articles, metrics, topic_summary, evidence, ai

articles, metrics, topic_summary, evidence, ai_assessments = load_data()

AI_FALLBACK = {
0: {'classification':'Emerging signal','decision':'Include','signal_name':'UK Sportswear Market Forecast 2032–2035','what_is_changing':'The AI interprets repeated market-size and forecast reports as evidence of continued growth in sportswear, activewear and premium outdoor apparel.','why_it_matters':'The AI suggests this could influence investment, innovation and market development in UK sportswear.','future_direction':'The AI expects further growth, premiumisation, innovation and sustainability activity.','evidence_limitations':'The evidence is mainly commercial market reports and may not reflect real consumer behaviour or nascent cultural change.','confidence':4},
1: {'classification':'Emerging signal','decision':'Include','signal_name':'UK Sportswear Trends and Consumer Behaviour Shift','what_is_changing':'The AI identifies changing consumer behaviour around sports-fashion partnerships, brand growth and evolving retail behaviour.','why_it_matters':'The AI suggests this could affect brand positioning, retail strategy and consumer preference.','future_direction':'The AI expects continued brand partnerships and more consumer-led sports-fashion strategies.','evidence_limitations':'Coverage has declined and the evidence does not yet prove a long-term shift.','confidence':3},
2: {'classification':'Emerging signal','decision':'Include','signal_name':'UK Retail & Consumer Trends 2026','what_is_changing':'The AI identifies broader retail and consumer behaviour changes, including digital shopping and evolving purchasing expectations.','why_it_matters':'The AI suggests sportswear brands may need more personalised digital retail and marketing strategies.','future_direction':'The AI expects a stronger digital retail presence and more personalised consumer engagement.','evidence_limitations':'The evidence is largely general retail intelligence rather than sportswear-specific consumer evidence.','confidence':4},
3: {'classification':'Emerging signal','decision':'Include','signal_name':'Nike Innovation and Expansion in UK Sportswear','what_is_changing':'The AI identifies Nike innovation, women’s apparel development, collaborations and strategic expansion.','why_it_matters':'The AI suggests these moves could influence competition, consumer behaviour and brand positioning.','future_direction':'The AI expects continued innovation, product launches and strategic partnerships.','evidence_limitations':'The evidence is concentrated around one brand and may not represent a wider market-level signal.','confidence':4},
4: {'classification':'Context','decision':'Exclude','signal_name':'AI Shopping & Future Consumer Context','what_is_changing':'The AI identifies AI-driven shopping, generational change and evolving fashion rules.','why_it_matters':'The AI suggests these changes may affect product design, digital engagement and consumer targeting.','future_direction':'The AI expects increased use of AI-driven retail and personalised consumer experiences.','evidence_limitations':'The evidence is mainly global fashion context and is not strongly grounded in UK sportswear.','confidence':2},
5: {'classification':'Emerging signal','decision':'Include','signal_name':'2026 Activewear Brand Focus','what_is_changing':'The AI interprets repeated “best activewear brand” coverage as growing interest in activewear brands.','why_it_matters':'The AI suggests this may indicate stronger consumer attention and brand activity in activewear.','future_direction':'The AI expects increased brand activity and potentially more entrants into the activewear market.','evidence_limitations':'Much of the evidence is editorial or affiliate-style content and coverage has declined strongly.','confidence':4},
6: {'classification':'Emerging signal','decision':'Include','signal_name':'2026 Wellness Trends in UK Sportswear','what_is_changing':'The AI identifies growing overlap between wellness, fitness, fashion and health-conscious consumer behaviour.','why_it_matters':'The AI suggests consumers may increasingly seek products that combine performance, comfort, wellbeing and aesthetics.','future_direction':'The AI expects more wellness-focused products, functional sportswear and fashion-forward performance design.','evidence_limitations':'The evidence is weighted toward wellness and fashion prediction content rather than direct sportswear consumer behaviour.','confidence':4}
}

VISUAL_DIRECTIONS = {
0:{'palette':[('Mineral Blue','#435D70'),('Moss','#66705A'),('Stone','#A69F91'),('Carbon','#303236')],'textures':['Bonded jersey','Weatherproof ripstop','Brushed technical fleece'],'product':'Premium multi-use outerwear, technical layering and durable commuter-performance pieces'},
1:{'palette':[('Cobalt','#3159A7'),('Graphite','#45464B'),('Optic White','#F2F1EC'),('Signal Lime','#98A957')],'textures':['Engineered mesh','Perforated knit','Four-way stretch woven'],'product':'Lifestyle-performance footwear, elevated tracksuits and sport-fashion crossover layers'},
2:{'palette':[('Digital Lavender','#8673A8'),('Ice','#C8D7DA'),('Silver','#8F969C'),('Ink','#252A32')],'textures':['Seamless knit','Reflective technical nylon','Lightweight mesh'],'product':'Digitally assisted shopping, fit-led footwear and personalised performance essentials'},
3:{'palette':[('Volt','#A5B936'),('Deep Black','#18191B'),('Sport Red','#A84B46'),('Cool Grey','#8A8E92')],'textures':['Compression knit','Laser-cut mesh','Bonded performance jersey'],'product':'Women’s performance systems, modular training layers and innovation-led footwear'},
4:{'palette':[('Chrome','#AAB1B8'),('Ultra Violet','#6F5B8E'),('Midnight','#202533'),('Glacier','#B7CBD0')],'textures':['Translucent technical mesh','Gloss-coated nylon','3D engineered knit'],'product':'AI-assisted product discovery, adaptive personalisation and data-informed fit'},
5:{'palette':[('Oxblood','#713D40'),('Espresso','#51443E'),('Slate','#646A70'),('Butter','#D6C899')],'textures':['Matte compression jersey','Soft rib knit','Peached technical fabric'],'product':'Elevated studio-to-street activewear, premium matching sets and refined everyday performance'},
6:{'palette':[('Sage','#879684'),('Mineral Pink','#B5908D'),('Oat','#C5BDAF'),('Ocean','#577786')],'textures':['Soft-touch rib','Air-knit jersey','Brushed recovery fabric'],'product':'Recovery layers, low-impact performance sets and comfort-led wellness apparel'}
}

def get_ai(topic_id):
    if not ai_assessments.empty:
        match = ai_assessments[ai_assessments['topic_id'] == topic_id]
        if not match.empty:
            return match.iloc[0].to_dict()
    return AI_FALLBACK.get(topic_id,{})

def get_topic(topic_id):
    metric = metrics[metrics['topic_id'] == topic_id].iloc[0]
    topic_articles = articles[articles['topic_id'] == topic_id].copy().sort_values('date', ascending=False)
    summary_match = topic_summary[topic_summary['Topic'] == topic_id]
    summary = summary_match.iloc[0] if not summary_match.empty else None
    topic_evidence = evidence[evidence['topic_id'] == topic_id].copy()
    if 'evidence_number' in topic_evidence.columns:
        topic_evidence = topic_evidence.sort_values('evidence_number')
    return metric, topic_articles, summary, topic_evidence

def clean_terms(summary):
    if summary is None:
        return []
    value = summary.get('Representation','')
    if not isinstance(value,str):
        return []
    value = value.replace('[','').replace(']','').replace("'",'')
    stop = {'the','and','of','in','to','is','how','by','for','with','on','are'}
    return [x.strip() for x in value.split(',') if x.strip() and x.strip().lower() not in stop][:6]

def swatches_html(topic_id):
    out = '<div class="swatch-row">'
    for name, colour in VISUAL_DIRECTIONS.get(topic_id,{}).get('palette',[]):
        out += f'<div class="swatch" style="background:{colour};">{name}</div>'
    return out + '</div>'

def participant_file():
    pid = st.session_state.participant_id.strip()
    return REVIEWS_FOLDER / f'{pid}_genx_stage2_reviews.csv'

def get_saved_reviews():
    f = participant_file()
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

def save_review(review):
    f = participant_file()
    new = pd.DataFrame([review])
    if f.exists():
        existing = pd.read_csv(f)
        existing = existing[existing['topic_id'] != review['topic_id']]
        new = pd.concat([existing,new], ignore_index=True)
    new.to_csv(f,index=False,encoding='utf-8-sig')

for k,v in {'started':False,'current_index':0,'participant_id':'','participant_role':'','years_experience':0}.items():
    if k not in st.session_state:
        st.session_state[k] = v

topic_ids = sorted(metrics['topic_id'].dropna().astype(int).unique().tolist())

if not st.session_state.started:
    st.markdown('''<div class="hero"><div class="eyebrow">Human–AI Foresight · Stage 2</div><h1 style="margin-bottom:8px;">Gen X × UK Sportswear</h1><p style="font-size:1.08rem;max-width:850px;">Review machine-detected evidence alongside the AI-generated interpretation, then assess what is genuinely relevant, culturally meaningful and strategically useful for Gen X consumers.</p></div>''', unsafe_allow_html=True)
    st.info('**Important:** The AI output shown here has not been corrected using previous professional feedback. This stage tests how professional judgement changes the AI interpretation when a defined Gen X consumer lens is applied.')
    st.markdown('<span class="tag">UK sportswear</span><span class="tag">Gen X</span><span class="tag">Technology</span><span class="tag">Consumer behaviour</span><span class="tag">12–24 months</span>', unsafe_allow_html=True)
    st.write('')
    c1,c2 = st.columns(2)
    with c1:
        participant_id = st.text_input('Participant ID', placeholder='e.g. P01')
        role = st.text_input('Professional role', placeholder='e.g. Foresight Strategist')
    with c2:
        experience = st.number_input('Years of relevant experience', min_value=0, max_value=50, value=0)
    with st.expander('What you are evaluating'):
        st.write('For each signal you will see the underlying evidence, machine-detected themes, source links, the AI-generated signal name and interpretation, the AI future direction and limitations, plus an exploratory colour / texture / product translation.')
        st.write('The visual direction is **not evidence**. It is included so you can judge whether generative AI turns weak intelligence into a persuasive-looking trend story, and whether those cues are meaningful for Gen X.')
    if st.button('Enter Gen X forecast review →', type='primary', use_container_width=True):
        if not participant_id.strip():
            st.error('Please enter a Participant ID.')
        else:
            st.session_state.participant_id = participant_id.strip()
            st.session_state.participant_role = role.strip()
            st.session_state.years_experience = experience
            st.session_state.started = True
            st.rerun()
    st.stop()

saved = get_saved_reviews()

if not saved.empty and saved['topic_id'].nunique() >= len(topic_ids):
    st.title('Gen X AI forecast review complete')
    accepted = saved[saved['professional_decision'].isin(['Accept','Accept with changes','Reframe'])]
    rejected = saved[saved['professional_decision'] == 'Reject']
    c1,c2,c3 = st.columns(3)
    c1.metric('Reviewed',len(saved)); c2.metric('Retained / reframed',len(accepted)); c3.metric('Rejected',len(rejected))
    st.divider()
    st.header('Overall AI forecast assessment')
    st.slider('How useful is this AI-generated forecast for professional Gen X foresight?',1,5,2,key='overall_usefulness')
    st.text_area('What are the biggest limitations of the AI forecast?',height=130,key='overall_limitations')
    st.text_area('Did the AI identify anything genuinely valuable?',height=120,key='valuable_elements')
    st.text_area('What important Gen X developments or opportunities are missing?',height=130,key='missing_genx')
    st.multiselect('What additional evidence would you need before considering this credible foresight?',['Reddit / online communities','Fan or enthusiast communities','Social media behaviour','Search behaviour','Consumer interviews','Retail observation','IRL events','Cultural media','Niche publications','Sales / market data','Product launches','Expert interviews','Other'],key='needed_sources')
    st.text_area('Any specific communities, publications, events or sources you would recommend?',height=120,key='source_notes')
    st.divider()
    st.download_button('Download Stage 2 review data', data=saved.to_csv(index=False).encode('utf-8'), file_name=f"{st.session_state.participant_id}_genx_stage2_reviews.csv", mime='text/csv')
    st.caption('Please email the downloaded CSV back to the researcher.')
    st.stop()

st.markdown('''<div class="hero"><div class="eyebrow">Stage 2 · AI interpretation review</div><h1 style="margin-bottom:6px;">Gen X Signal Studio</h1><p>Compare the underlying evidence with the AI forecast interpretation and visual direction.</p></div>''', unsafe_allow_html=True)

dashboard_columns = st.columns(min(len(topic_ids),4))
for index,topic_id in enumerate(topic_ids):
    metric,_,_,_ = get_topic(topic_id)
    ai = get_ai(topic_id)
    with dashboard_columns[index % len(dashboard_columns)]:
        st.markdown(f"**{index+1:02d} · {ai.get('signal_name',f'Signal {index+1}')}**")
        st.caption(f"{int(metric['total_articles'])} articles · {int(metric['unique_sources'])} sources · AI confidence {ai.get('confidence','—')}/5")

st.divider()
current_index = st.session_state.current_index
topic_id = topic_ids[current_index]
metric, topic_articles, summary, topic_evidence = get_topic(topic_id)
ai = get_ai(topic_id)
terms = clean_terms(summary)
visual = VISUAL_DIRECTIONS.get(topic_id,{})
st.progress((current_index+1)/len(topic_ids))
st.caption(f'Signal {current_index+1} of {len(topic_ids)}')
st.title(ai.get('signal_name',f'Possible Signal {current_index+1}'))
st.caption('AI-generated signal name shown alongside the original machine-detected cluster')

evidence_col, ai_col = st.columns([1,1])
with evidence_col:
    st.subheader('1 · Underlying signal evidence')
    if terms:
        st.write('**Machine-detected themes:** ' + ' · '.join(terms))
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Articles',int(metric['total_articles']))
    c2.metric('Sources',int(metric['unique_sources']))
    c3.metric('Active months',int(metric['active_months']))
    growth = metric['growth_percent']
    c4.metric('Coverage','N/A' if pd.isna(growth) else f'{growth:+.0f}%')
    st.markdown('<div class="small">These metrics describe the evidence cluster. They do not prove that it is an emerging trend.</div>', unsafe_allow_html=True)

with ai_col:
    st.subheader('2 · AI interpretation')
    st.markdown(f'''<div class="ai-box"><div class="eyebrow">AI classification</div><strong>{ai.get('classification','—')}</strong> · Decision: <strong>{ai.get('decision','—')}</strong> · Confidence: <strong>{ai.get('confidence','—')}/5</strong></div>''', unsafe_allow_html=True)
    st.markdown('**What the AI thinks is changing**'); st.write(ai.get('what_is_changing',''))
    st.markdown('**Why the AI thinks it matters**'); st.write(ai.get('why_it_matters',''))
    st.markdown('**AI future direction**'); st.write(ai.get('future_direction',''))
    with st.expander("AI's own evidence limitations"):
        st.write(ai.get('evidence_limitations',''))

st.divider(); st.subheader('3 · Representative evidence')
st.caption('Open the original links if you want to investigate the evidence in more depth.')
for _,row in topic_evidence.iterrows():
    title = row.get('title','Untitled'); source = row.get('source','Unknown source'); date = row.get('date')
    date_text = date.strftime('%d %b %Y') if pd.notna(date) else 'Unknown date'
    with st.expander(title):
        st.caption(f'{source} · {date_text}')
        text = row.get('available_text','')
        if isinstance(text,str) and text.strip(): st.write(text[:700])
        url = row.get('url','')
        if isinstance(url,str) and url.strip(): st.markdown(f'[Open original article ↗]({url})')

with st.expander(f'Browse all {len(topic_articles)} items in this cluster'):
    for _,row in topic_articles.iterrows():
        st.markdown(f"**{row.get('title','Untitled')}**")
        date = row.get('date'); date_text = date.strftime('%d %b %Y') if pd.notna(date) else ''
        st.caption(f"{row.get('source','')} · {date_text}")
        url = row.get('url','')
        if isinstance(url,str) and url.strip(): st.markdown(f'[Open article]({url})')
        st.divider()

st.divider(); st.subheader('4 · AI visual translation')
st.markdown('<div class="context-box"><strong>Exploratory, not evidence.</strong><br>These colour, texture and product cues are a speculative visual translation of the AI-generated signal. Please judge whether they are meaningful, stereotyped, irrelevant or useful for a Gen X sportswear consumer.</div>', unsafe_allow_html=True)
st.markdown('**Colour direction**'); st.markdown(swatches_html(topic_id), unsafe_allow_html=True)
st.markdown('**Texture / material direction**')
texture_cols = st.columns(3)
for index,texture in enumerate(visual.get('textures',[])):
    with texture_cols[index % 3]:
        st.markdown(f'<div class="texture"><div class="eyebrow">Material cue</div><strong>{texture}</strong></div>', unsafe_allow_html=True)
st.markdown('**Possible product expression**'); st.write(visual.get('product',''))

st.divider(); st.header('5 · Professional Gen X review')
st.write('Assess the AI output specifically through a Gen X lens. You do not need to preserve the AI framing.')
genx_relevance = st.slider('How relevant is this signal to Gen X consumers?',1,5,2,key=f'genx_relevance_{topic_id}')
signal_maturity = st.radio('For Gen X, how would you classify this?',['Emerging opportunity','Growing / gaining relevance','Already established','Relevant context only','Weak / unsupported','Not relevant'],index=None,key=f'maturity_{topic_id}')
evidence_grounding = st.slider('How well is the AI interpretation grounded in the evidence?',1,5,2,key=f'grounding_{topic_id}')
cultural_accuracy = st.slider('How well does it understand Gen X cultural / consumer context?',1,5,2,key=f'culture_{topic_id}')
strategic_usefulness = st.slider('How strategically useful could this be for a sportswear client?',1,5,2,key=f'usefulness_{topic_id}')
visual_usefulness = st.slider('How useful is the AI visual direction for Gen X?',1,5,2,key=f'visual_{topic_id}')
professional_decision = st.radio('Professional decision',['Accept','Accept with changes','Reframe','Reject'],index=None,horizontal=True,key=f'professional_decision_{topic_id}')
what_ai_missed = st.text_area('What has the AI misunderstood, oversimplified or missed?',height=120,key=f'missed_{topic_id}')
genx_reframe = st.text_area('How would you reframe this specifically for Gen X?',height=120,key=f'reframe_{topic_id}')
client_value = st.text_area('What would make this useful for NPD, marketing, consumer targeting or creative direction?',height=120,key=f'client_value_{topic_id}')
visual_feedback = st.text_area('Would you change the colours, textures or product expression? If yes, how?',height=110,key=f'visual_feedback_{topic_id}')
missing_evidence = st.multiselect('What evidence is missing?',['Gen X consumer interviews','Reddit / community discussion','Social media groups','IRL observation / events','Retail behaviour','Sales data','Search data','Niche cultural media','Product launches','Historical context','Competitor activity','Other'],key=f'missing_evidence_{topic_id}')
confidence = st.slider('Confidence in your professional judgement',1,5,4,key=f'confidence_{topic_id}')

st.divider(); back_col,next_col = st.columns([1,3])
with back_col:
    if current_index > 0 and st.button('← Previous'):
        st.session_state.current_index -= 1; st.rerun()
with next_col:
    if st.button('Save & Next →',type='primary',use_container_width=True):
        if signal_maturity is None:
            st.error('Please classify the signal for Gen X.')
        elif professional_decision is None:
            st.error('Please choose a professional decision.')
        else:
            review = {
                'participant_id':st.session_state.participant_id,
                'participant_role':st.session_state.participant_role,
                'years_experience':st.session_state.years_experience,
                'topic_id':topic_id,
                'ai_signal_name':ai.get('signal_name',''),
                'ai_classification':ai.get('classification',''),
                'ai_decision':ai.get('decision',''),
                'ai_confidence':ai.get('confidence',''),
                'genx_relevance':genx_relevance,
                'genx_signal_maturity':signal_maturity,
                'evidence_grounding':evidence_grounding,
                'cultural_accuracy':cultural_accuracy,
                'strategic_usefulness':strategic_usefulness,
                'visual_usefulness':visual_usefulness,
                'professional_decision':professional_decision,
                'what_ai_missed':what_ai_missed,
                'genx_reframe':genx_reframe,
                'client_value':client_value,
                'visual_feedback':visual_feedback,
                'missing_evidence':' | '.join(missing_evidence),
                'professional_confidence':confidence,
                'machine_total_articles':metric['total_articles'],
                'machine_unique_sources':metric['unique_sources'],
                'machine_growth_percent':metric['growth_percent'],
                'timestamp':datetime.now().isoformat()
            }
            save_review(review)
            if current_index < len(topic_ids)-1:
                st.session_state.current_index += 1
            st.rerun()
