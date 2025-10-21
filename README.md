# 🏥 Medical NLP Pipeline

This project implements an end-to-end **Medical NLP Pipeline** that processes doctor–patient transcripts to extract relevant medical information, summarize it, and analyze intent and sentiment.

---

## 🚀 Features

The pipeline performs **three major NLP tasks**:

### **Task 1: Named Entity Recognition (NER)**
- Extracts key entities from medical transcripts such as:
  - Symptoms
  - Diagnosis
  - Treatment
  - Dates
  - Prognosis
- Combines **spaCy**'s pretrained medical NER model with **custom rule-based heuristics** for better accuracy on unstructured text.

### **Task 2: Summarization and Keyword Extraction**
- Generates a concise summary of the transcript using **transformer-based models (BART or T5)**.
- Extracts top medical and contextual keywords using **KeyBERT** (fallback to TF-IDF when embeddings are unavailable).

### **Task 3: Sentiment, Intent Analysis & SOAP Note Generation**
- Performs sentiment and intent analysis using heuristic keyword mapping (optionally supports ClinicalBERT fine-tuning).
- Generates **SOAP Notes** (Subjective, Objective, Assessment, Plan) using structured output from previous tasks.

---

## 🧠 Approach

The project follows a **multi-stage NLP pipeline** approach:

1. **Preprocessing** — Cleans and tokenizes input text using spaCy pipelines.
2. **NER Extraction** — Uses pretrained models + custom rules to capture domain-specific entities.
3. **Keyword Extraction** — Applies embedding-based similarity ranking (KeyBERT).
4. **Summarization** — Leverages a transformer model to generate short summaries.
5. **Sentiment & Intent Detection** — Uses lexicon-based scoring and keyword analysis.
6. **SOAP Note Generation** — Structures final output into medical documentation format.

---

## ⚙️ Installation & Setup

### **1. Clone the repository**
```bash
git clone https://github.com/rishee10/Physician-Notetaker
cd Physician-Notetaker
```

### **2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### **4. Run the pipeline**
```bash
python main.py
```

---

## 📊 Directory Structure
```
medical-nlp-pipeline/
│
├── main.py     # Main pipeline script
├── requirements.txt            # Dependencies
├── README.md                   # Documentation               
```


