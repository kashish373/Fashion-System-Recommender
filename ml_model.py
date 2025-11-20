from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import joblib
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

# Default model path inside Flask instance folder
MODEL_PATH = os.path.join('instance', 'model.joblib')

# ---- Utilities to turn wardrobe items into text ----
@dataclass
class SimpleItem:
    category: Optional[str] = None
    pattern: Optional[str] = None
    material: Optional[str] = None
    fit: Optional[str] = None
    seasonality: Optional[str] = None
    color_hex: Optional[str] = None
    occasions: Optional[List[str]] = None


def item_to_text(i: SimpleItem) -> str:
    parts = [
        i.category or '',
        i.pattern or '',
        i.material or '',
        i.fit or '',
        i.seasonality or '',
        (i.color_hex or '').replace('#', 'hex'),
        ' '.join(i.occasions or []),
    ]
    return ' '.join([p.lower() for p in parts if p])


def outfit_to_text(items: List[SimpleItem], context: Dict) -> str:
    ctx = [
        (context.get('occasion') or ''),
        (context.get('weather') or ''),
        (context.get('time') or '')
    ]
    joined_items = ' '.join(item_to_text(i) for i in items)
    return (joined_items + ' ' + ' '.join(ctx)).strip().lower()


# ---- Training / Loading ----

def _build_regression_pipeline() -> Pipeline:
    # Single text column named 'text'
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    pipe = Pipeline([
        ('tfidf', vectorizer),
        ('rf', model)
    ])
    return pipe


def _build_classification_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    pipe = Pipeline([
        ('tfidf', vectorizer),
        ('rf', model)
    ])
    return pipe


def train_from_csv(csv_path: str) -> Dict:
    """Train a model from a CSV. Expected columns:
    - If doing regression: must contain a numeric column named 'score'
    - If doing classification: must contain a binary/multiclass column named 'label'
    - If neither is present, we fall back to unsupervised: we fit a TF-IDF vectorizer on available textual columns and store it.
    - Feature columns (any subset): category, pattern, material, fit, seasonality, color_hex, occasions, productDisplayName, articleType, subCategory, baseColour, usage, season, gender, and optionally a free-form 'text' column
    We will concatenate available columns into a single 'text' feature. The trained model is saved to MODEL_PATH.
    Returns metrics (if supervised) and info about the task type.
    """
    df = pd.read_csv(csv_path)

    # Decide problem type
    task: Optional[str] = None
    target_col: Optional[str] = None
    if 'score' in df.columns:
        # Attempt regression
        if pd.api.types.is_numeric_dtype(df['score']):
            task = 'regression'
            target_col = 'score'
    if task is None and 'label' in df.columns:
        task = 'classification'
        target_col = 'label'

    # Feature columns from both wardrobe-style and catalog-style datasets
    candidate_text_cols = [
        'text', 'category', 'pattern', 'material', 'fit', 'seasonality', 'color_hex', 'occasions',
        'productDisplayName', 'articleType', 'subCategory', 'baseColour', 'usage', 'season', 'gender'
    ]
    text_cols = [c for c in candidate_text_cols if c in df.columns]
    if not text_cols:
        raise ValueError("Dataset must include at least one textual column like category, articleType, baseColour, productDisplayName, or a 'text' column")

    # Build text feature from known columns or a pre-existing 'text'
    df['__text__'] = (
        df[text_cols]
        .astype(str)
        .apply(lambda row: ' '.join(row.values.tolist()), axis=1)
        .str.replace('#', 'hex', regex=False)
        .str.lower()
    )

    X = df['__text__']

    if task is None:
        # Unsupervised: fit only a TF-IDF vectorizer and store it
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        vectorizer.fit(X)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"task": "unsupervised", "vectorizer": vectorizer}, MODEL_PATH)
        return {"task": "unsupervised", "metrics": None, "model_path": MODEL_PATH, "num_rows": int(len(df))}
    else:
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if task == 'regression':
            pipe = _build_regression_pipeline()
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))
            metrics = {"mae": mae, "r2": r2}
        else:
            pipe = _build_classification_pipeline()
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, average='weighted'))
            metrics = {"accuracy": acc, "f1": f1}

        # Ensure instance dir exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"task": task, "pipeline": pipe}, MODEL_PATH)

        return {"task": task, "metrics": metrics, "model_path": MODEL_PATH}


def load_model():
    if os.path.exists(MODEL_PATH):
        obj = joblib.load(MODEL_PATH)
        return obj
    return None


def score_outfit(items: List[SimpleItem], context: Dict, model_obj) -> float:
    """Score an outfit using the trained model object {task, pipeline}.
    Returns a higher-is-better score in [0,1] for classification (probability of positive if available), or a raw regression output scaled using a simple sigmoid-like transform for display.
    """
    if not model_obj:
        return 0.0
    text = outfit_to_text(items, context)
    task = model_obj['task']
    if task == 'unsupervised':
        # Use the vectorizer to compute average pairwise cosine similarity among items and with context tokens
        vectorizer: TfidfVectorizer = model_obj['vectorizer']
        # Split outfit into item texts to compute pairwise similarities
        item_texts = [item_to_text(i) for i in items]
        texts = item_texts + [text]
        X = vectorizer.transform(texts)
        sim = cosine_similarity(X)
        # Average pairwise similarity among items (upper triangle)
        n = len(item_texts)
        if n <= 1:
            intra = 0.5
        else:
            vals = []
            for a in range(n):
                for b in range(a+1, n):
                    vals.append(sim[a, b])
            intra = float(sum(vals) / len(vals)) if vals else 0.5
        # Similarity of outfit aggregate text to context-enhanced text (last row/col)
        ctx_sim = float(sim[-1, :-1].mean()) if n > 0 else 0.5
        # Weighted score
        return 0.7 * intra + 0.3 * ctx_sim
    elif task == 'classification':
        # If classifier supports predict_proba use it, else use decision_function
        pipe: Pipeline = model_obj['pipeline']
        if hasattr(pipe, 'predict_proba'):
            import numpy as np
            proba = pipe.predict_proba([text])
            if proba.shape[1] == 2:
                return float(proba[0, 1])
            else:
                # Multi-class: return max probability
                return float(proba.max())
        else:
            # Fallback to label prediction mapped to 0/1
            pred = pipe.predict([text])[0]
            try:
                return float(pred)
            except Exception:
                return 0.0
    else:
        # Regression: scale to 0..1 using 1 / (1 + exp(-x)) with mild factor
        import math
        pipe: Pipeline = model_obj['pipeline']
        val = float(pipe.predict([text])[0])
        return 1.0 / (1.0 + math.exp(-0.1 * val))
