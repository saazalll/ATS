# AI-Powered Resume Analyzer (Open Source)

A modern Streamlit ATS dashboard that analyzes resumes against job descriptions using only open-source Python libraries (no OpenAI API, no paid APIs).

## Core Features

- Upload resume(s): **PDF, DOCX, TXT**
- Single and bulk resume screening
- TF-IDF + cosine similarity ATS scoring
- Skill extraction from local JSON database
- Matched / missing skills analysis
- Downloadable PDF report

## Advanced ATS Features

- Keyword density analysis (JD skills in resume)
- Skill heatmap visualization (Plotly)
- Resume readability score (Flesch)
- Experience level estimator (Entry / Mid / Senior)
- Section detection (Skills / Education / Projects / Experience)
- Ranking score breakdown chart
- Matched keyword highlighting in resume preview
- Smart improvement suggestions based on gaps

## Modern UI

- Wide layout + tab-based navigation
- Dark glassmorphism-inspired custom CSS
- KPI cards (ATS score, matched, missing)
- Animated progress bars
- SaaS-style spacing/typography and leaderboard view

## Project Structure

```text
/app.py
/modules/
    parser.py
    cleaner.py
    scorer.py
    skill_extractor.py
    report_generator.py
    advanced_analysis.py
/data/skill_database.json
requirements.txt
README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## NLTK Troubleshooting (punkt_tab)

If you see errors like `Resource punkt_tab not found`, run:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

The app also includes a fallback tokenizer if punkt resources are missing.
