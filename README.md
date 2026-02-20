# AI-Powered Resume Analyzer (Open Source)

A modern Streamlit web app that analyzes resumes against a job description using open-source Python libraries only (no OpenAI API, no paid APIs, no cloud dependency required).

## Features

- Upload resume(s): **PDF, DOCX, TXT**
- Paste a job description
- Resume text extraction (pdfplumber, python-docx)
- NLP preprocessing (NLTK):
  - lowercase normalization
  - stopword removal
  - lemmatization
- Skill extraction using a local predefined skill database (`data/skill_database.json`)
- ATS similarity scoring using **TF-IDF + cosine similarity** (scikit-learn)
- Insightful results:
  - Match score (%)
  - Matched skills
  - Missing skills
  - Resume strength analysis
  - Weakness areas
- Data visualizations using Plotly:
  - Matched vs Missing skills
  - Similarity score
  - Bulk ranking chart
- Bulk screening for multiple resumes with ranking
- Download detailed PDF report (reportlab)
- Clean UI with sidebar navigation

---

## Project Structure

```text
/app.py
/modules/
    parser.py
    cleaner.py
    scorer.py
    skill_extractor.py
    report_generator.py
/data/skill_database.json
requirements.txt
README.md
```

---

## Installation

### 1) Clone repository

```bash
git clone <your-repo-url>
cd ATS
```

### 2) Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the app

```bash
streamlit run app.py
```

Then open your browser at the URL shown by Streamlit (typically `http://localhost:8501`).

---

## How It Works

1. Resume parser reads PDF/DOCX/TXT and extracts plain text.
2. Text cleaner preprocesses resume and job description for robust comparison.
3. Skill extractor finds known skills from local JSON database.
4. Scorer computes ATS similarity with TF-IDF and cosine similarity.
5. Results page displays score, skill gaps, strengths/weaknesses, and charts.
6. PDF generator creates a downloadable candidate report.

---

## Notes

- This project is fully local-first and API-free.
- To improve skill matching quality, expand `data/skill_database.json` with domain-specific skills.
- For production deployment, consider adding authentication and persistent storage.

