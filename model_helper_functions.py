#majority of these generated using Claude
#https://claude.ai/share/6bf771bb-f0ad-4e05-98c3-122e7ae544c5

#save_model_results runs evaluate_classifier and saves it to a csv file
#evaluate_classifier is a helper function that computes various metrics for a fitted sklearn classifier

#export_model_results_to_markdown converts the csv file to markdown files which can easily be mapped to a table, with Obsidian in mind

#run for fit and valid data, just name such that _fit and _valid are in the model name

#note: this is completely untested

#ToDo:
#create obsidian file + folder
#add additional result visualizations
#function to add custom hyperparameter fields to model md results or model-specific csv, map out flow
#view_model_results is a helper function that loads and displays the results from the csv file, unpacking matrices as needed
#convert_to_markdown is a meta function that takes saved model results and puts them in a markdown table, with the confusion matrix and get_class fully separated


import pandas as pd
import json
import os


###
# data cleaning functions
###

# convert categorical variables in df to dummy variables
def cat_to_dummy(df,numericize=True):
    spectral = pd.get_dummies(df['spectral_type'], dtype=int).astype("category")
    galaxy_pop = pd.get_dummies(df['galaxy_population'], dtype=int).astype("category")
    
    # Drop original categorical columns and concatenate dummy variables
    new_df = df.drop(columns=['spectral_type', 'galaxy_population'])
    new_df = pd.concat([new_df, spectral, galaxy_pop], axis=1)
    new_df['spectral_type'] = df['spectral_type'].astype("category")
    new_df['galaxy_population'] = df['galaxy_population'].astype("category")

    class_mapping = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
    if('class' in new_df.columns):
        new_df['class'] = new_df['class'].map(class_mapping)
    #reclassify all as numeric, may need to roll this back depending on if models treat as categorical or not
    if(numericize):
        new_df = new_df.apply(pd.to_numeric)
    
    return new_df

# convert the dummy numbers back into the categories
def dummy_to_cat(df):
    class_mapping = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}
    df['class'] = df['class'].map(class_mapping)
    return df

def feature_engineer(df):
    # apply data transformations and feature engineering
    # convert alpha and delta to radians, then use cos() and sin() to map from the 2D plane to a 3D sphere
    alpha_rad = np.radians(df['alpha'])
    delta_rad = np.radians(df['delta'])

    df['X_loc'] = np.cos(delta_rad) * np.cos(alpha_rad)
    df['Y_loc'] = np.cos(delta_rad) * np.sin(alpha_rad)
    df['Z_loc'] = np.sin(delta_rad)

    df = df.drop(columns=['alpha', 'delta'])

    # adding in new features that are common subtractions of the different 'ugriz' filters

    df['u_g_diff'] = df['u'] - df['g']
    df['g_r_diff'] = df['g'] - df['r']
    df['r_i_diff'] = df['r'] - df['i']
    df['i_z_diff'] = df['i'] - df['z']

    return df


###
# model-saving functions
###

def save_model_results(
    model,
    X,
    y_true,
    model_name: str,
    model_description: str = "",
    labels=None,
    csv_path: str = "raw_model_results.csv",
    output_dir = "model_outputs/"
):
    """
    Evaluate a fitted sklearn classifier and persist results to a CSV.

    Calls evaluate_classifier() internally, then appends a row to
    ``csv_path`` (creating it with headers if it doesn't exist yet).

    Parameters
    ----------
    model            : fitted sklearn classifier
    X                : feature matrix (array-like)
    y_true           : ground-truth labels (array-like)
    model_name       : short identifier stored in the CSV
    model_description: optional free-text description (default "")
    labels           : optional list of class labels; passed to evaluate_classifier
    csv_path         : path to the CSV file (default "raw_model_results.csv")

    Returns
    -------
    dict – the full metrics dict returned by evaluate_classifier
    """
    metrics = evaluate_classifier(model, X, y_true, labels=labels)

    # ── Flatten metrics into a single-level dict ─────────────────────────────
    # per_class and confusion_matrix are complex structures; serialise them to
    # JSON strings so they survive a CSV round-trip cleanly.
    row = {
        "model_name":        model_name,
        "model_description": model_description,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_precision":   metrics["macro_precision"],
        "macro_recall":      metrics["macro_recall"],
        "macro_f1":          metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall":   metrics["weighted_recall"],
        "weighted_f1":       metrics["weighted_f1"],
        "matthews_corrcoef": metrics["matthews_corrcoef"],
        "cohen_kappa":       metrics["cohen_kappa"],
        "roc_auc_ovr":       metrics["roc_auc_ovr"],
        # Nested structures stored as JSON strings
        "per_class":         json.dumps(metrics["per_class"]),
        "confusion_matrix":  json.dumps(metrics["confusion_matrix"].tolist()),
    }

    new_row_df = pd.DataFrame([row])

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        updated_df = new_row_df

    updated_df.to_csv(csv_path, index=False)

    pd.DataFrame(metrics["y_pred"]).to_csv(output_dir + model_name + ".csv", index=False)

    return metrics

from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
import numpy as np


def evaluate_classifier(model, X, y_true, labels=None):
    """
    Evaluate a fitted sklearn classifier and return a comprehensive metrics dict.

    Parameters
    ----------
    model    : fitted sklearn classifier
    X        : feature matrix (array-like)
    y_true   : ground-truth labels (array-like)
    labels   : optional list of class labels in desired order;
               defaults to sorted unique values in y_true

    Returns
    -------
    dict with keys:
        balanced_accuracy       – average per-class recall (imbalance-robust)
        per_class               – dict keyed by label, each with precision /
                                  recall / f1 / support
        macro_precision         – unweighted mean precision across classes
        macro_recall            – unweighted mean recall across classes
        macro_f1                – unweighted mean F1 across classes
        weighted_precision      – precision weighted by class support
        weighted_recall         – recall weighted by class support
        weighted_f1             – F1 weighted by class support
        matthews_corrcoef       – MCC: single [-1, 1] correlation coefficient
        cohen_kappa             – agreement above chance
        roc_auc_ovr             – one-vs-rest ROC-AUC (requires predict_proba)
        confusion_matrix        – raw confusion matrix (numpy array)
    """
    if labels is None:
        labels = sorted(set(y_true))

    y_pred = model.predict(X)

    # ── Per-class metrics ────────────────────────────────────────────────────
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {
        label: {
            "precision": float(p),
            "recall":    float(r),
            "f1":        float(f),
            "support":   int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }

    # ── Aggregate metrics ────────────────────────────────────────────────────
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # ── ROC-AUC (one-vs-rest, needs probability estimates) ──────────────────
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X)
            multi_class = "ovr" if len(labels) > 2 else "raise"
            roc_auc = float(
                roc_auc_score(y_true, y_prob, multi_class=multi_class, labels=labels)
                if len(labels) > 2
                else roc_auc_score(y_true, y_prob[:, 1])
            )
        except Exception:
            roc_auc = None  # e.g. only one class present in y_true

    return {
        # ── Imbalance-robust overall accuracy ────────────────────────────
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),

        # ── Per-category breakdown ────────────────────────────────────────
        "per_class": per_class,

        # ── Macro averages (treat every class equally) ────────────────────
        "macro_precision": float(macro_p),
        "macro_recall":    float(macro_r),
        "macro_f1":        float(macro_f),

        # ── Weighted averages (weight by class frequency) ─────────────────
        "weighted_precision": float(weighted_p),
        "weighted_recall":    float(weighted_r),
        "weighted_f1":        float(weighted_f),

        # ── Single-value multi-class summaries ────────────────────────────
        # MCC: best single-number summary for imbalanced multi-class problems
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        # Cohen's Kappa: measures agreement beyond what chance would produce
        "cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        # ROC-AUC (None if model has no predict_proba)
        "roc_auc_ovr":       roc_auc,

        # ── Raw confusion matrix ──────────────────────────────────────────
        "confusion_matrix":  confusion_matrix(y_true, y_pred, labels=labels),
        "y_pred" : y_pred
    }


###
# code to map to markdown constituent files
###

import pandas as pd
import json
import os
import re


def _confusion_matrix_to_latex(matrix: list[list[int]]) -> str:
    """Convert a 2-D list into a LaTeX pmatrix block for rendering in markdown."""
    rows = " \\\\\n    ".join(
        " & ".join(str(cell) for cell in row)
        for row in matrix
    )
    return f"$$\n\\begin{{pmatrix}}\n    {rows}\n\\end{{pmatrix}}\n$$"


def _sanitise_filename(name: str) -> str:
    """Strip characters that are unsafe in filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def export_model_results_to_markdown(
    csv_path: str = "raw_model_results.csv",
    output_dir: str = ".",
):
    """
    Read raw_model_results.csv and write one markdown file per row.

    Each file is named ``<model_name>.md`` and contains every metric on its
    own line in the format ``variable_name:: value``.

    ``per_class`` is unpacked so each class's precision / recall / f1 /
    support appears as its own line, e.g.::

        per_class.cat.precision:: 0.91

    ``confusion_matrix`` is rendered as a LaTeX matrix block.

    Parameters
    ----------
    csv_path   : path to the CSV produced by save_model_results()
    output_dir : directory to write the markdown files into (default: cwd)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find results file: {csv_path!r}")

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for _, row in df.iterrows():
        lines = [
            f"# Model Report: {row['model_name']}",
            "",
        ]

        filename = _sanitise_filename(str(row["model_name"])) + ".md"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"Skipped (already exists): {filepath}")
            continue

        for col, value in row.items():

            # ── per_class: unpack each class → metric as its own line ────────
            if col == "per_class":
                per_class: dict = json.loads(value)
                for label, class_metrics in per_class.items():
                    for metric_name, metric_value in class_metrics.items():
                        lines.append(
                            f"per_class.{label}.{metric_name}:: {metric_value}"
                        )

            # ── confusion_matrix: render as LaTeX ────────────────────────────
            elif col == "confusion_matrix":
                matrix: list[list[int]] = json.loads(value)
                lines.append("confusion_matrix::")
                lines.append("")
                lines.append(_confusion_matrix_to_latex(matrix))

            # ── all other scalars ─────────────────────────────────────────────
            else:
                lines.append(f"{col}:: {value}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Written: {filepath}")

