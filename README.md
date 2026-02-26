# ATS Resume Analyzer (Open Source)

Modern Streamlit ATS platform with attractive UI, light/dark theme toggle, and 4 working modules:
1. Single resume analysis
2. Bulk resume analysis (up to 500 files)
3. Video resume analysis (video + transcript)
4. Resume builder

## What it does

- ATS score for single and bulk resumes
- Missing skills extraction
- Bulk shortlisting based on ATS threshold
- Shortlisted results table with:
  - ATS score
  - Confidence score (SVM-based)
- Bulk charts:
  - ATS score vs file name
  - ATS score vs confidence (grouped bar)
- CSV download for shortlisted and all analyzed results
- SVM-based confidence prediction to complement keyword matching
- PDF report download for each candidate

## Advanced ATS Insights

- Keyword density analysis
- Skill heatmap visualization
- Resume readability score (Flesch)
- Experience level estimation (Entry/Mid/Senior)
- Section detection (Skills/Education/Projects/Experience)
- Ranking score breakdown chart
- Matched keyword highlighting in resume preview
- Smart improvement suggestions

## Tech Stack

- Python 3.10+
- Streamlit
- scikit-learn (TF-IDF + cosine + SVM)
- NLTK
- pdfplumber
- python-docx
- plotly
- reportlab

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
pip install --upgrade pip
pip install -r requirements.txt
```

## UI Highlights

- Super-modern glassmorphism dashboard
- Sidebar light/dark mode switch
- Animated KPI cards and progress bars
- Structured tabs for single/bulk/video/builder workflows

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## NLTK Troubleshooting

If you get `punkt_tab` errors:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

The app includes fallback tokenization/stopword logic as a backup.
