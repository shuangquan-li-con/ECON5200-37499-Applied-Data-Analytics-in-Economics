📊 FedSpeak NLP Diagnostics & Predictive Analysis
📌 Project Overview

This project builds and diagnoses a full Natural Language Processing (NLP) pipeline for analyzing Federal Reserve (FOMC) meeting minutes. The goal is to identify common modeling mistakes, correct them using domain-appropriate methods, and evaluate whether textual signals can predict future monetary policy decisions.

The workflow moves from debugging a flawed pipeline to building a reusable module and comparing modern text representations in a predictive setting.

⚙️ What I Did
1. Diagnosed a Broken NLP Pipeline

I identified three critical issues in the original implementation:

Naive tokenization (whitespace-only splitting)
Incorrect sentiment dictionary (general-purpose instead of finance-specific)
Poor TF-IDF configuration (no document-frequency filtering or n-grams)

Each issue materially affected downstream results and interpretability.

2. Rebuilt the Pipeline (Production-Oriented)

I implemented a corrected NLP workflow:

Proper preprocessing using word_tokenize, regex cleaning, stopword removal, and lemmatization
Domain-specific sentiment analysis using a simplified Loughran–McDonald dictionary
Improved TF-IDF feature engineering with:
min_df=5
max_df=0.85
bigrams (ngram_range=(1,2))
3. Compared Text Representations (Clustering)

I evaluated two approaches:

TF-IDF (sparse, frequency-based)
Sentence-Transformer embeddings (dense, semantic)

Using K-Means and silhouette scores, I compared how well each representation captures document similarity.

4. Predictive Modeling (Time Series Setup)

I built a time-aware prediction task:

Target: whether the next FOMC meeting occurs during a tightening regime
Method:
TimeSeriesSplit (expanding window)
Logistic Regression
AUC-ROC evaluation

To ensure robustness, I handled edge cases where some folds contained only one class.

5. Modularized the Pipeline

I converted the workflow into a reusable Python module:

src/fomc_sentiment.py

The module includes:

preprocess_fomc() — text cleaning pipeline
compute_lm_sentiment() — financial sentiment scoring
build_tfidf_matrix() — feature construction

This reflects a more production-ready design compared to notebook-only code.

📈 Key Findings
Embeddings vs TF-IDF:
Embeddings generally capture broader semantic structure, while TF-IDF performs well when policy language is repetitive and structured.
Predictive Signal:
Text alone contains some signal about future policy regimes, but results are sensitive to time splits and class imbalance.
Pipeline Design Matters:
Small preprocessing or feature engineering mistakes can significantly distort results in NLP workflows.
🧠 What This Project Demonstrates
Ability to debug real-world ML pipelines, not just build them
Understanding of domain-specific NLP (finance/economics)
Experience with time-series model evaluation (no leakage)
Writing clean, reusable, modular Python code
Translating technical work into interpretable insights
