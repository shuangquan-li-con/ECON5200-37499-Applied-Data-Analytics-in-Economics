"""
fomc_sentiment.py — FOMC Text Analysis Module

Reusable functions for preprocessing, sentiment scoring, and
TF-IDF vectorization of Federal Reserve meeting minutes.

Author: Shuangquan Li
Course: ECON 5220, Lab 23
"""

import re
from typing import List, Tuple, Dict

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Simplified Loughran-McDonald dictionaries
LM_NEGATIVE = {
   'adverse', 'adversely', 'concern', 'concerned', 'concerns',
   'decline', 'declined', 'declining', 'decrease', 'decreased',
   'deficit', 'deteriorate', 'deteriorated', 'deteriorating',
   'difficult', 'difficulty', 'downturn', 'fail', 'failure',
   'falling', 'loss', 'losses', 'negative', 'negatively',
   'recession', 'recessionary', 'risk', 'risks', 'risky',
   'severe', 'severely', 'slowdown', 'sluggish', 'stress',
   'stressed', 'threat', 'threaten', 'troubled', 'uncertain',
   'uncertainty', 'unfavorable', 'volatile', 'volatility',
   'vulnerability', 'vulnerable', 'weak', 'weaken',
   'weakened', 'weakness', 'worse', 'worsen', 'worsened'
}

LM_POSITIVE = {
   'achieve', 'achieved', 'achievement', 'benefit', 'beneficial',
   'confidence', 'confident', 'favorable', 'gain', 'gained',
   'gains', 'good', 'growth', 'improve', 'improved',
   'improvement', 'improving', 'increase', 'increased',
   'opportunity', 'optimism', 'optimistic', 'positive',
   'positively', 'progress', 'rebound', 'recover',
   'recovery', 'strength', 'strengthen', 'strong',
   'stronger', 'success', 'successful'
}

LM_UNCERTAINTY = {
   'approximate', 'approximately', 'assume', 'assumption',
   'believe', 'cautious', 'could', 'depend', 'depends',
   'doubt', 'estimate', 'expect', 'expected', 'forecast',
   'indefinite', 'likelihood', 'may', 'might', 'perhaps',
   'possible', 'possibly', 'predict', 'preliminary',
   'probable', 'probably', 'risk', 'roughly', 'seem',
   'suggest', 'tentative', 'uncertain', 'uncertainty',
   'unclear', 'unpredictable', 'variable'
}


def preprocess_fomc(text: str) -> str:
   """
   Clean and preprocess FOMC text.

   Steps:
   1. Lowercase
   2. Tokenize with NLTK word_tokenize
   3. Remove non-alphabetic characters
   4. Remove stopwords and very short tokens
   5. Lemmatize

   Args:
       text: Raw text string

   Returns:
       Cleaned text as a single string
   """
   stop_words = set(stopwords.words('english'))
   lemmatizer = WordNetLemmatizer()

   text = text.lower()
   tokens = word_tokenize(text)

   clean_tokens = []
   for tok in tokens:
       tok = re.sub(r'[^a-z]', '', tok)
       if tok and tok not in stop_words and len(tok) > 2:
           tok = lemmatizer.lemmatize(tok)
           clean_tokens.append(tok)

   return ' '.join(clean_tokens)


def compute_lm_sentiment(text: str) -> Dict[str, float]:
   """
   Compute simplified Loughran-McDonald sentiment metrics.

   Args:
       text: Preprocessed text string

   Returns:
       Dictionary with sentiment counts and ratios
   """
   tokens = text.split()
   total = len(tokens)

   if total == 0:
       return {
           'neg_count': 0,
           'pos_count': 0,
           'uncertainty_count': 0,
           'neg_ratio': 0.0,
           'pos_ratio': 0.0,
           'uncertainty_ratio': 0.0,
           'net_sentiment': 0.0
       }

   neg_count = sum(1 for t in tokens if t in LM_NEGATIVE)
   pos_count = sum(1 for t in tokens if t in LM_POSITIVE)
   uncertainty_count = sum(1 for t in tokens if t in LM_UNCERTAINTY)

   return {
       'neg_count': neg_count,
       'pos_count': pos_count,
       'uncertainty_count': uncertainty_count,
       'neg_ratio': neg_count / total,
       'pos_ratio': pos_count / total,
       'uncertainty_ratio': uncertainty_count / total,
       'net_sentiment': (pos_count - neg_count) / total
   }


def build_tfidf_matrix(
   texts: List[str],
   min_df: int = 5,
   max_df: float = 0.85,
   max_features: int = 5000
) -> Tuple:
   """
   Build a TF-IDF matrix from preprocessed FOMC texts.

   Args:
       texts: List of cleaned text documents
       min_df: Minimum document frequency
       max_df: Maximum document frequency
       max_features: Maximum number of features

   Returns:
       Tuple of (tfidf_matrix, vectorizer)
   """
   vectorizer = TfidfVectorizer(
       min_df=min_df,
       max_df=max_df,
       max_features=max_features,
       ngram_range=(1, 2)
   )

   tfidf_matrix = vectorizer.fit_transform(texts)
   return tfidf_matrix, vectorizer


# Quick self-test
if __name__ == "__main__":
   sample_text = "The committee noted that inflation risks remain elevated, but labor market conditions improved."
   clean_text = preprocess_fomc(sample_text)
   scores = compute_lm_sentiment(clean_text)
   tfidf_matrix, vectorizer = build_tfidf_matrix(
   [clean_text], 
   min_df=1, 
   max_df=1.0
)

   print("Clean text:", clean_text)
   print("Sentiment scores:", scores)
   print("TF-IDF shape:", tfidf_matrix.shape)
   print("Module ran successfully.")
