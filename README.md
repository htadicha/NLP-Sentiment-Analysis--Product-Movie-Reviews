# NLP-Sentiment-Analysis--Product-Movie-Reviews

Purpose
- Perform exploratory data analysis, preprocessing, feature extraction and classification/segmentation on product/movie/customer reviews (NLP sentiment & segmentation).

Repository layout
- data/                       — CSV dataset(s). Default expected: data/customer_reviews.csv
- segementation.ipynb         — main analysis notebook (loads data, preprocessing, modeling).
- scripts/                    — utility scripts (create as needed).
- README.md                   — this file.

Quick notes about the notebook (segementation.ipynb)
- Key variables and behavior found in the notebook:
  - DATA_PATH = "data/customer_reviews.csv"
  - load_data(DATA_PATH): tries pd.read_csv(DATA_PATH). If not found, it searches the data/ folder and returns the first CSV it finds; raises FileNotFoundError otherwise.
  - Current cell loads data with: df = pd.read_csv(DATA_PATH)
    - Recommendation: replace that line with df = load_data(DATA_PATH) to enable the fallback lookup behavior defined in the notebook.
- Important imports used: pandas, numpy, sklearn.model_selection.train_test_split, sklearn.ensemble.RandomForestClassifier, sklearn.metrics.classification_report, sklearn.feature_extraction.text.TfidfTransformer

Requirements
- Ubuntu 24.04.2 LTS dev container.
- Python 3.10+ recommended.
- Install required packages:
  - pip install --user pandas numpy scikit-learn jupyterlab matplotlib seaborn nltk

Run the notebook (dev container)
1. From workspace root open Jupyter Lab:
   - jupyter lab --ip=0.0.0.0 --no-browser --allow-root
2. Open the notebook via VS Code Jupyter integration or in the browser.
3. To open the Jupyter URL in the host browser from the container:
   - $BROWSER <url>

Data expectations
- The notebook expects a CSV with at least:
  - A text column (commonly named: text, review, comment)
  - A label column for supervised tasks (commonly named: label, sentiment, rating)
- If column names differ, either rename columns or adapt notebook cells to point to the correct column names.
- For large files, use pd.read_csv(..., chunksize=...) or switch to Polars/Dask.

Suggested analysis pipeline
1. Data loading & validation (use load_data)
2. Basic cleaning:
   - lowercase, remove URLs, punctuation, extra whitespace
   - handle missing values
3. Tokenization / optional lemmatization (NLTK / spaCy)
4. Feature extraction:
   - TF-IDF (TfidfVectorizer/TfidfTransformer), word n-grams
   - optionally embeddings (word2vec/transformers) for advanced models
5. Modeling:
   - simple baseline: LogisticRegression or RandomForest
   - advanced: transformers (fine-tune), stacking
6. Evaluation:
   - train/test split (use random_state and stratify where applicable)
   - accuracy, classification_report, confusion_matrix, ROC/AUC (binary/multi)
7. Experiment logging:
   - record dataset snapshot, parameters, seed, and metrics

Reproducibility
- Set random_state on train_test_split and model constructors.
- Save vectorizer and model artifacts (joblib.dump).

Troubleshooting
- FileNotFoundError for DATA_PATH:
  - Update DATA_PATH in segementation.ipynb to the correct CSV, or
  - Use the provided load_data function and ensure data/ contains a CSV.
- Missing packages:
  - Activate the same Python interpreter used by Jupyter/VS Code and install packages there.
- If notebook cells fail on memory, reduce dataset size or use chunked reads.

Recommended edits (small)
- Replace the explicit load line in segementation.ipynb:
  - df = pd.read_csv(DATA_PATH)
  - with:
  - df = load_data(DATA_PATH)
- Add simple text-cleaning utility and use TfidfVectorizer instead of raw TfidfTransformer pipeline for convenience.

Next steps / enhancements
- Add a scripts/runner.py to run a reproducible experiment end-to-end.
- Add unit tests for data loading and cleaning functions.
- Add example small CSV in data/ (non-sensitive) to make the repo runnable without external data.

Contact / notes
- Keep sensitive data out of the repository. Commit only sanitized sample data and code.
```// filepath: /workspaces/NLP-Sentiment-Analysis--Product-Movie-Reviews/README.md
# NLP-Sentiment-Analysis--Product-Movie-Reviews

Purpose
- Perform exploratory data analysis, preprocessing, feature extraction and classification/segmentation on product/movie/customer reviews (NLP sentiment & segmentation).

Repository layout
- data/                       — CSV dataset(s). Default expected: data/customer_reviews.csv
- segementation.ipynb         — main analysis notebook (loads data, preprocessing, modeling).
- scripts/                    — utility scripts (create as needed).
- README.md                   — this file.

Quick notes about the notebook (segementation.ipynb)
- Key variables and behavior found in the notebook:
  - DATA_PATH = "data/customer_reviews.csv"
  - load_data(DATA_PATH): tries pd.read_csv(DATA_PATH). If not found, it searches the data/ folder and returns the first CSV it finds; raises FileNotFoundError otherwise.
  - Current cell loads data with: df = pd.read_csv(DATA_PATH)
    - Recommendation: replace that line with df = load_data(DATA_PATH) to enable the fallback lookup behavior defined in the notebook.
- Important imports used: pandas, numpy, sklearn.model_selection.train_test_split, sklearn.ensemble.RandomForestClassifier, sklearn.metrics.classification_report, sklearn.feature_extraction.text.TfidfTransformer

Requirements
- Ubuntu 24.04.2 LTS dev container.
- Python 3.10+ recommended.
- Install required packages:
  - pip install --user pandas numpy scikit-learn jupyterlab matplotlib seaborn nltk

Run the notebook (dev container)
1. From workspace root open Jupyter Lab:
   - jupyter lab --ip=0.0.0.0 --no-browser --allow-root
2. Open the notebook via VS Code Jupyter integration or in the browser.
3. To open the Jupyter URL in the host browser from the container:
   - $BROWSER <url>

Data expectations
- The notebook expects a CSV with at least:
  - A text column (commonly named: text, review, comment)
  - A label column for supervised tasks (commonly named: label, sentiment, rating)
- If column names differ, either rename columns or adapt notebook cells to point to the correct column names.
- For large files, use pd.read_csv(..., chunksize=...) or switch to Polars/Dask.

Suggested analysis pipeline
1. Data loading & validation (use load_data)
2. Basic cleaning:
   - lowercase, remove URLs, punctuation, extra whitespace
   - handle missing values
3. Tokenization / optional lemmatization (NLTK / spaCy)
4. Feature extraction:
   - TF-IDF (TfidfVectorizer/TfidfTransformer), word n-grams
   - optionally embeddings (word2vec/transformers) for advanced models
5. Modeling:
   - simple baseline: LogisticRegression or RandomForest
   - advanced: transformers (fine-tune), stacking
6. Evaluation:
   - train/test split (use random_state and stratify where applicable)
   - accuracy, classification_report, confusion_matrix, ROC/AUC (binary/multi)
7. Experiment logging:
   - record dataset snapshot, parameters, seed, and metrics

Reproducibility
- Set random_state on train_test_split and model constructors.
- Save vectorizer and model artifacts (joblib.dump).

Troubleshooting
- FileNotFoundError for DATA_PATH:
  - Update DATA_PATH in segementation.ipynb to the correct CSV, or
  - Use the provided load_data function and ensure data/ contains a CSV.
- Missing packages:
  - Activate the same Python interpreter used by Jupyter/VS Code and install packages there.
- If notebook cells fail on memory, reduce dataset size or use chunked reads.

Recommended edits (small)
- Replace the explicit load line in segementation.ipynb:
  - df = pd.read_csv(DATA_PATH)
  - with:
  - df = load_data(DATA_PATH)
- Add simple text-cleaning utility and use TfidfVectorizer instead of raw TfidfTransformer pipeline for convenience.

Next steps / enhancements
- Add a scripts/runner.py to run a reproducible experiment end-to-end.
- Add unit tests for data loading and cleaning functions.
- Add example small CSV in data/ (non-sensitive) to make the repo runnable without external data.

Contact / notes
- Keep sensitive data out of the repository. Commit only sanitized sample data and code.