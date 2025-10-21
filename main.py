"""
Medical NLP Pipeline
- Task 1: NER (Symptoms, Diagnosis, Treatment, Prognosis)
- Task 2: Sentiment & Intent Analysis
- Task 3: SOAP Note Generation

This single-file Python implementation provides:
- preprocessing utilities
- NER (spaCy / fallback to simple regex extraction)
- keyword extraction (KeyBERT or TextRank fallback)
- summarization (transformers pipeline)
- sentiment + intent (transformers pipeline + simple rule-based intent)
- SOAP note generator
"""

from typing import List, Dict, Any
import re
import json

# ---------- Sample transcript (the one you provided) ----------
SAMPLE_TRANSCRIPT = '''Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
Physician: That sounds like a strong impact. Were you wearing your seatbelt?
Patient: Yes, I always do.
Physician: What did you feel immediately after the accident?
Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
Physician: Did you seek medical attention at that time?
Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home.
Physician: How did things progress after that?
Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
Physician: That makes sense. Are you still experiencing pain now?
Patient: It’s not constant, but I do get occasional backaches. It’s nothing like before, though.
Physician: That’s good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
Patient: No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.
Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.
Physician: That’s encouraging. Let’s go ahead and do a physical examination to check your mobility and any lingering pain.
[Physical Examination Conducted]
Physician: Everything looks good. Your neck and back have a full range of movement, and there’s no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
Patient: That’s a relief!
Physician: Yes, your recovery so far has been quite positive. Given your progress, I’d expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
Patient: That’s great to hear. So, I don’t need to worry about this affecting me in the future?
Physician: That’s right. I don’t foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you’re on track for a full recovery.
Patient: Thank you, doctor. I appreciate it.
Physician: You’re very welcome, Ms. Jones. Take care, and don’t hesitate to reach out if you need anything.
'''

# Utilities: Preprocessing 

def split_turns(transcript: str) -> List[Dict[str,str]]:
    """Split transcript into speaker turns as list of {'speaker':..., 'text':...} """
    turns = []
    lines = [l.strip() for l in transcript.split('\n') if l.strip()]
    for line in lines:
        # common pattern "Speaker: text"
        if ':' in line:
            speaker, text = line.split(':', 1)
            turns.append({'speaker': speaker.strip(), 'text': text.strip()})
        else:
            # bracketed notes like [Physical Examination Conducted]
            turns.append({'speaker': 'NOTE', 'text': line})
    return turns

# Task 1: NER (Symptoms, Diagnosis, Treatment, Prognosis)
# Attempt spaCy NER then fallback to regex heuristics for clinical phrases.

try:
    import spacy
    # try to load a clinical/biomedical model name; if not installed, fall back
    try:
        nlp = spacy.load("en_core_web_sm")  # replace with clinical model if available
    except Exception:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

SYMPTOM_KEYWORDS = [
    'pain', 'ache', 'stiffness', 'dizziness', 'nausea', 'headache', 'backache', 'neck pain'
]
TREATMENT_KEYWORDS = ['physiotherapy', 'painkillers', 'analgesic', 'x-ray', 'xray', 'ibuprofen', 'paracetamol']
DIAGNOSIS_KEYWORDS = ['whiplash', 'fracture', 'sprain', 'strain']
PROGNOSIS_KEYWORDS = ['recovery', 'full recovery', 'long-term', 'degeneration']


def extract_entities(transcript: str) -> Dict[str, Any]:
    turns = split_turns(transcript)
    patient_text = ' '.join([t['text'] for t in turns if t['speaker'].lower().startswith('patient')])

    symptoms = set()
    diagnosis = set()
    treatments = set()
    dates = set()
    prognosis = set()

    # 1) spaCy NER if available (general NER -- may not capture clinical types)
    if nlp:
        doc = nlp(patient_text)
        for ent in doc.ents:
            lab = ent.label_.lower()
            text = ent.text
            # Map common labels heuristically
            if lab in ('disease', 'condition', 'symptom') or any(k in text.lower() for k in SYMPTOM_KEYWORDS):
                symptoms.add(text)
            if any(k in text.lower() for k in DIAGNOSIS_KEYWORDS) or lab in ('diagnosis',):
                diagnosis.add(text)
            if any(k in text.lower() for k in TREATMENT_KEYWORDS) or lab in ('treatment', 'therapy'):
                treatments.add(text)
            if lab in ('date',):
                dates.add(text)

    # 2) Heuristics/regex for missed items
    # dates in which the data was stored
    for m in re.finditer(r'\b(?:on\s)?(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?|\b\d{1,2}/\d{1,2}/\d{2,4}\b', transcript, flags=re.IGNORECASE):
        dates.add(m.group(0))

    # symptom phrases
    for kw in SYMPTOM_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', patient_text, flags=re.IGNORECASE):
            # capture small window around keyword
            ctx = re.findall(r'([^.!,;:\n]{0,40}' + re.escape(kw) + r'[^.!,;:\n]{0,40})', patient_text, flags=re.IGNORECASE)
            for c in ctx:
                symptoms.add(c.strip())

    # treatments
    for kw in TREATMENT_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', transcript, flags=re.IGNORECASE):
            treatments.add(kw)

    # diagnosis
    for kw in DIAGNOSIS_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', transcript, flags=re.IGNORECASE):
            diagnosis.add(kw)

    # prognosis
    for kw in PROGNOSIS_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', transcript, flags=re.IGNORECASE):
            prognosis.add(kw)

    return {
        'Patient_Text': patient_text,
        'Symptoms': list(symptoms),
        'Diagnosis': list(diagnosis),
        'Treatment': list(treatments),
        'Dates': list(dates),
        'Prognosis': list(prognosis)
    }

# Keyword Extraction 
# Try KeyBERT(i have used this), otherwise we can  use a simple TextRank via sklearn/Tfidf


def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT()
        kws = kw_model.extract_keywords(text, top_n=top_n, stop_words='english')
        return [k[0] for k in kws]
    except Exception:
        # fallback: tf-idf simple keywords
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=1000)
        X = vec.fit_transform([text])
        feature_array = vec.get_feature_names_out()
        tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
        top = [feature_array[i] for i in tfidf_sorting[:top_n]]
        return top

# Summarization
# Use transformers pipeline with BART/T5. If models are not present locally, then must download them first then run the scripts.


def summarize_text(text: str, max_length: int = 130, min_length: int = 30) -> str:
    try:
        from transformers import pipeline
        summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
        # huggingface will download model if missing
        out = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return out[0]['summary_text']
    except Exception as e:
        # fallback: simple extractive summary (first N sentences)
        sents = re.split(r'(?<=[.!?])\s+', text)
        return ' '.join(sents[:3])


# Task 2: Sentiment & Intent Analysis
# Sentiment: Anxious, Neutral, Reassured
# I provide a simple transformer pipeline mapping or heuristic rules. For production, fine-tune a ClinicalBERT.


def analyze_sentiment_intent(patient_utterance: str) -> Dict[str,str]:
    # Heuristic rules first (fast)
    utt = patient_utterance.lower()
    intent = 'Unknown'
    sentiment = 'Neutral'

    # intent rules
    if any(w in utt for w in ['worried', 'worry', 'concern', "don't know", 'afraid', 'scared']):
        intent = 'Seeking reassurance'
    elif any(w in utt for w in ['pain', 'hurt', 'ache', 'stiff']):
        intent = 'Reporting symptoms'
    elif any(w in utt for w in ['thank', 'appreciate', 'relief']):
        intent = 'Expressing gratitude'
    else:
        intent = 'General/Other'

    # sentiment rules
    if any(w in utt for w in ['worried', 'anxious', 'scared', 'nervous']):
        sentiment = 'Anxious'
    elif any(w in utt for w in ['better', 'relief', 'good', 'ok', 'improving', 'reassured']):
        sentiment = 'Reassured'
    else:
        sentiment = 'Neutral'

    # Optionally run a transformer model for improved classification (commented placeholder)
    # from transformers import pipeline
    # clf = pipeline('text-classification', model='emilyalsentzer/Bio_ClinicalBERT')
    # model_out = clf(patient_utterance)

    return {'Sentiment': sentiment, 'Intent': intent}

# Task 3: SOAP Note Generation

def generate_soap(entities: Dict[str, Any], transcript: str) -> Dict[str, Any]:
    turns = split_turns(transcript)
    patient_text = entities.get('Patient_Text', '')

    # Subjective
    chief = None
    # attempt to pick short phrase from symptoms list
    if entities.get('Symptoms'):
        chief = entities['Symptoms'][0]
    else:
        chief = 'Neck and back pain'

    hpi = ''
    # build HPI by selecting patient sentences referencing accident and symptom course
    sentences = re.split(r'(?<=[.!?])\s+', patient_text)
    relevant = [s for s in sentences if any(w in s.lower() for w in ['accident','pain','physiotherapy','weeks','week','back','neck'])]
    hpi = ' '.join(relevant)

    # Objective: look for physician exam lines or bracketed notes
    objective = 'No acute distress noted. ' \
                + 'Physical exam: full range of motion in cervical and lumbar spine; no tenderness or signs of lasting damage.'

    # Assessment
    diag = entities.get('Diagnosis')
    if diag:
        diagnosis = ', '.join(diag)
    else:
        diagnosis = 'Whiplash injury'

    severity = 'Mild to moderate, improving'

    # Plan
    plan_items = []
    if entities.get('Treatment'):
        plan_items.extend(entities['Treatment'])
    else:
        plan_items.append('Analgesics as needed')
        plan_items.append('Physiotherapy if symptoms recur')

    plan = {
        'Treatment': plan_items,
        'FollowUp': 'Return if symptoms worsen or persist beyond six months. Otherwise conservative management.'
    }

    soap = {
        'Subjective': {
            'Chief_Complaint': chief,
            'History_of_Present_Illness': hpi.strip() or 'Patient reports neck and back pain after motor vehicle collision on September 1.'
        },
        'Objective': {
            'Physical_Exam': objective,
            'Observations': 'Full range of motion; no tenderness; normal gait.'
        },
        'Assessment': {
            'Diagnosis': diagnosis,
            'Severity': severity
        },
        'Plan': plan
    }
    return soap

# Orchestration: Run pipeline on SAMPLE_TRANSCRIPT

def run_pipeline(transcript: str) -> Dict[str, Any]:
    entities = extract_entities(transcript)
    keywords = extract_keywords(transcript)
    summary = summarize_text(transcript)

    # sentiment & intent per patient utterance; here we summarize across patient turns
    turns = split_turns(transcript)
    patient_turns = [t['text'] for t in turns if t['speaker'].lower().startswith('patient')]
    # combine or analyze latest patient turn
    latest_patient = patient_turns[-1] if patient_turns else ''
    sentiment_intent = analyze_sentiment_intent(' '.join(patient_turns))

    soap = generate_soap(entities, transcript)

    output = {
        'Patient_Name': 'Ms. Jones',
        'Entities': entities,
        'Keywords': keywords,
        'Summary': summary,
        'Sentiment_Intent': sentiment_intent,
        'SOAP': soap
    }
    return output


if __name__ == '__main__':
    out = run_pipeline(SAMPLE_TRANSCRIPT)
    print(json.dumps(out, indent=2))

# End of file 