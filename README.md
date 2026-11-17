# Stellar Classification: Galaxy, Quasar or Star

Classifies sky-survey objects as `GALAXY`, `QSO` (quasar) or `STAR` from photometry, redshift and sky position. Built for the [Kaggle Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6/overview) competition, which is scored on **balanced accuracy** (the unweighted mean of per-class recall).

This project was completed as a joint effort between Ryan Halstater (rmh6247 (at) psu (dot) edu) and Steven Liang (steven (dot) liang1999 (at) gmail (dot) com)

**Headline result:** CatBoost reaches a validation balanced accuracy of **0.948**. Redshift is by far the most influential feature, followed by the engineered color indices. Categorical features are the least influential.

## Results at a glance

All scores are on a 25% hold-out (144,336 rows) that the models never trained on.

| Model | Configuration | Balanced accuracy | Accuracy | Macro F1 | MCC | ROC AUC (OvR) |
|---|---|---|---|---|---|---|
| **CatBoost** | depth 6, learning rate 0.5, 100 iterations | **0.9483** | 0.9626 | 0.9492 | 0.9266 | 0.9954 |
| XGBoost | depth 6, learning rate 1, 9 trees | 0.9439 | 0.9596 | 0.9449 | 0.9205 | 0.9948 |
| Multinomial logistic regression | best of the runs tried | 0.9142 | 0.8927 | 0.8675 | 0.8117 | 0.9812 |

## Data

- **Source:** Kaggle Playground Series S6E6, related to the earlier [SDSS17 stellar classification dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17).
- **Size:** 577,347 training rows, no missing values.
- **Target:** `class`, with three values.

| Feature | Type | Meaning |
|---|---|---|
| `alpha`, `delta` | float | Right ascension and declination: the object's longitude and latitude on the sky, in degrees |
| `u`, `g`, `r`, `i`, `z` | float | Brightness in five filters from ultraviolet to infrared, on the magnitude scale (lower = brighter) |
| `redshift` | float | How far the object's light is shifted by its motion and by cosmic expansion. Slightly negative values are valid and typical of nearby stars |
| `Spectral_Type` | categorical | `M`, `G/K`, `O/B`, `A/F` |
| `Galaxy_Population` | categorical | `Red_Sequence`, `Blue_Cloud` |

## Pipeline

### 1. Cleaning

Rows with `u < 10` are dropped for being scientifically infeasible, which is confirmed by no such rows appearing in the testing dataset.

### 2. Feature engineering

Feature engineering lives in `model_helper_functions.feature_engineer`.

**Sky position to 3D coordinates.** Variables representing angles (`alpha` and `delta`) mapped onto sphere, and their (x,y,z) coordinates are used as features to allow treating 359° and 1° as similar.

```
X_loc = cos(delta) * cos(alpha)
Y_loc = cos(delta) * sin(alpha)
Z_loc = sin(delta)
```

**Color indices.** Scientifically motivated: galaxies, quasars and stars spread their light differently across the five filters, so the differences between neighboring bands separate them better than the raw brightness values do:

```
u_g_diff = u - g
g_r_diff = g - r
r_i_diff = r - i
i_z_diff = i - z
```

**Categoricals.** `Spectral_Type` and `Galaxy_Population` are one-hot encoded (`model_helper_functions.cat_to_dummy`) into `A/F`, `G/K`, `M`, `O/B`, `Blue_Cloud` and `Red_Sequence`. The target is encoded as 0 = `GALAXY`, 1 = `QSO`, 2 = `STAR`.

The final model sees 19 features: `redshift`, five magnitudes, four color indices, three position coordinates and six indicator columns.

### 3. Validation

A random 75/25 split (`train_test_split`, `random_state=12345`, not stratified) gives 433,008 training rows and 144,336 hold-out rows. Every run calls `model_helper_functions.save_model_results`, which appends the same set of metrics to `raw_model_results.csv` for both splits: balanced accuracy, macro and weighted precision, recall and F1, Matthews correlation, Cohen's kappa, one-vs-rest ROC AUC, per-class scores and the confusion matrix.

The final model is picked by ranking the hold-out rows on balanced accuracy and checking the top candidates against their training-set scores.

## Models

### Multinomial logistic regression

Achieves 0.914 balanced accuracy with class-specific accuracy issues (star precision of 0.670).

### XGBoost

Hyperparameter tuning via a grid search of 180 configurations: `max_depth` 2–6, `n_estimators` 1–9, `learning_rate` 0.01, 0.1, 0.5 or 1.

### CatBoost

Hyperparameter tuning via a grid search of 20 configurations that are close to those of XGBoost: `depth` 2–6 by `learning_rate` 0.01, 0.1, 0.5 or 1, each with 100 iterations. The six indicator columns are passed to CatBoost as categorical features.

Depth 6 with learning rate 0.5 is the best model, and the top three hold-out scores across every logged run are all CatBoost. As with XGBoost, the best depth is the largest one tried.

## Best model in detail

`CatBoostClassifier(iterations=100, learning_rate=0.5, depth=6)`, scored on the hold-out set:

| Class | Precision | Recall | F1 | Rows |
|---|---|---|---|---|
| `GALAXY` | 0.9731 | 0.9748 | 0.9740 | 94,578 |
| `QSO` | 0.9613 | 0.9621 | 0.9617 | 28,999 |
| `STAR` | 0.9162 | 0.9078 | 0.9120 | 20,759 |

Confusion matrix (rows are the true class, columns the predicted class):

| | Predicted `GALAXY` | Predicted `QSO` | Predicted `STAR` |
|---|---|---|---|
| **True `GALAXY`** | 92,197 | 934 | 1,447 |
| **True `QSO`** | 822 | 27,901 | 276 |
| **True `STAR`** | 1,723 | 190 | 18,846 |

Stars are the hardest class, as the exploratory analysis predicted. Of the 5,392 hold-out errors, 59% are galaxies and stars mistaken for each other, 33% are galaxy and quasar mix-ups, and only 9% are quasar and star mix-ups.

## Feature Importance Via SHAP

SHAP values are a common tool for determining feature importance for an arbitrary model. 
They were computed on the hold-out set for the best model. Reported below is the mean absolute SHAP value of each feature over the three classes.
come straight from CatBoost, computed on the hold-out set (`X_test`) for the best model:

| Rank | Feature | Mean \|SHAP\| | Share of total |
|---|---|---|---|
| 1 | `redshift` | 1.570 | 35.3% |
| 2 | `g_r_diff` | 0.386 | 8.7% |
| 3 | `r_i_diff` | 0.372 | 8.4% |
| 4 | `u_g_diff` | 0.320 | 7.2% |
| 5 | `z` | 0.311 | 7.0% |
| 6 | `u` | 0.221 | 5.0% |
| 7 | `g` | 0.184 | 4.1% |
| 8 | `X_loc` | 0.169 | 3.8% |
| 9 | `Z_loc` | 0.165 | 3.7% |
| 10 | `Y_loc` | 0.156 | 3.5% |
| 11 | `i_z_diff` | 0.148 | 3.3% |
| 12 | `r` | 0.143 | 3.2% |
| 13 | `i` | 0.119 | 2.7% |
| 14 | `Blue_Cloud` | 0.068 | 1.5% |
| 15 | `M` | 0.057 | 1.3% |
| 16 | `A/F` | 0.019 | 0.4% |
| 17 | `O/B` | 0.016 | 0.4% |
| 18 | `G/K` | 0.015 | 0.3% |
| 19 | `Red_Sequence` | 0.014 | 0.3% |

Summed by feature group:

| Group | Features | Share of total |
|---|---|---|
| Redshift | 1 | 35.3% |
| Color indices | 4 | 27.5% |
| Raw magnitudes | 5 | 22.0% |
| Sky position | 3 | 11.0% |
| Spectral type | 4 | 2.4% |
| Galaxy population | 2 | 1.9% |

Key points:

- **Redshift dominates.** Its importance is about four times that of the next feature and a third of the total. This matches the exploratory finding that high redshift marks quasars, and the physics: stars sit near zero, galaxies further out, quasars furthest.
- **The color indices earned their place.** Three of the four engineered colors rank second to fourth, ahead of every raw magnitude.
- **The categorical features add little.** Exploratory analysis flagged `Galaxy_Population` and `Spectral_Type` as good indicators, yet together they account for about 4%. The likely reason is that the colors already carry the same information

## Discussion

Through this analysis we found a model that does fairly well at predicting stellar object classification. There are several avenues for improvement available:

- Increase depth and number of trees: optimal values occured near limits, so it is possible performance would continue to increase. limits were chosen for computational purposes.
- Focus on the galaxy and star classification confusion, possibly leveraging ensemble approaches
- Use SHAP for model selection in a stepwise fashion
- Expand model class to other random forests and neural network approaches

## Repository layout

| Path | Contents |
|---|---|
| `EDA.ipynb` | Exploratory analysis: per-class distributions, sky-position plots |
| `data_notebook.ipynb` | Preprocessing, the three model families, model selection, test predictions, SHAP |
| `model_helper_functions.py` | `feature_engineer`, `cat_to_dummy`, `dummy_to_cat`, `save_model_results` |
| `raw_model_results.csv` | One row per model run and split, with all logged metrics |
| `data/` | `train.csv` and `test.csv` from Kaggle |
| `submissions/` | `final_predictions.csv` |



