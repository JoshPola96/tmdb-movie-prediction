<!-- README.md -->
# Predicting a film's rating before release

> **Scope** · Coursework from my Data Science & ML internship at Irohub Infotech (2024–25). Built quickly and published on GitHub on 2025-04-20, it was not revisited until this audit and rebuild as a tutorial in October 2026.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoshPola96/tmdb-movie-prediction/blob/main/tmdb_movies_analysis_prediction.ipynb)
Can a film's audience rating be predicted from what is known before release: budget, runtime, genres, language, release date, director and lead actor? This repository is one tutorial notebook that answers the question on the TMDB 5000 dataset without leakage, compares twelve model families fairly, and reports the result with its uncertainty.

## What it teaches

| Section | Concept |
|---|---|
| 4 | The prediction moment: deciding which columns a model may see (target leakage) |
| 5 | Sentinel values and unit errors in real data |
| 6 | Shrinkage: the Bayesian average as a ranking tool, not a prediction target |
| 7 | Splitting by time; survivorship bias in a "top 5,000" dataset |
| 8 | One leak-free preprocessing pipeline: multi-hot genres, cross-fitted target encoding |
| 9–10 | Forward-chaining cross-validation across twelve model families; the one-standard-error rule |
| 11 | Testing once, with bootstrap confidence intervals; diagnosing concept drift |
| 12 | Rating bands: rounding a regression versus a direct classifier; macro-F1 |
| 13 | Permutation importance, and what a negative importance means |
| 14 | Saving a model with skops instead of pickle |
| 15 | The first version's mistakes, each with the check that catches it |

## Results

The test set is the 567 films released from 2013 onwards with at least 50 votes. It was scored once, after every modelling choice had been made on earlier films.

| Model | Test MAE (rating points, 0–10 scale) | 95% CI |
|---|---|---|
| Linear regression on pre-release features | 0.619 | 0.580–0.659 |
| Baseline: predict the training mean | 0.703 | |

- The model beats the baseline by 0.084 points (95% CI 0.054–0.115): a real but modest gain.
- Cross-validation on earlier films estimated 0.530. The gap is **concept drift**: the correlation between budget and rating was −0.21 before 2013 and +0.17 after.
- Predictions spread much less than real ratings (standard deviation 0.39 against 0.87). Runtime and genre carry most of the signal; the model captures broad tendencies and cannot pick out an outstanding film.
- SVR had the best tuned cross-validation score (0.517), but within one standard error of plain linear regression, so the simpler model was chosen. Director and lead actor improved cross-validation by less than its noise and were left out by the same rule.
- Low / Mid / High rating bands: macro-F1 0.482 against 0.333 for chance. A direct logistic classifier beat rounding the regression's predictions into bands.

## Corrections to the first version

The first version reported near-perfect scores. Each one was an artifact:

| Reported | Cause | Now |
|---|---|---|
| Regression MSE 0.0001 | The target was the weighted rating, and its own inputs (`vote_average`, `vote_count`) were features | Pre-release features only: MAE 0.619 |
| Classification accuracy 1.0000 | The label `rating_class` was one of the features | Macro-F1 0.482 on later films |
| Both | Genres were exploded into rows before a random split, so 86% of test rows were films also in training | One row per film; split by release date |
| Model ranking | Nine regressors and five classifiers were tuned and ranked on the test set | Compared on training folds; tested once |

Section 15 of the notebook lists every mistake with the check that catches it.

## Run it

**In the browser:** use the Colab badge above. The notebook installs what Colab lacks and reads the data from this repository.

**Locally**, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/JoshPola96/tmdb-movie-prediction.git
cd tmdb-movie-prediction
uv sync
```

Open `tmdb_movies_analysis_prediction.ipynb` in VS Code, select the `.venv` kernel and run all cells; or run `uv run --with jupyterlab jupyter lab`. Versions are pinned in `uv.lock`.

## Repository contents

| Path | What it is |
|---|---|
| `tmdb_movies_analysis_prediction.ipynb` | The tutorial, with outputs |
| `data/` | The two TMDB CSVs |
| `models/tmdb_rating_model.skops` | The trained model; section 14 shows how to load it safely |
| `pyproject.toml`, `uv.lock` | Dependencies, pinned |

## Data and licence

The data is the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle, version 2), built from The Movie Database API. Its use is governed by [TMDB's terms](https://www.themoviedb.org/api-terms-of-use).

<a href="https://www.themoviedb.org/"><img src="https://www.themoviedb.org/assets/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg" alt="TMDB logo" width="120"></a>

*This product uses TMDB and the TMDB APIs but is not endorsed, certified, or otherwise approved by TMDB.*

The code is MIT-licensed; see [LICENSE](LICENSE).

## References

- Kapoor, S. & Narayanan, A. (2023). *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns 4(9).
- Breiman, L., Friedman, J., Olshen, R. & Stone, C. (1984). *Classification and Regression Trees.* (the one-standard-error rule)
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., §7.10.

The notebook lists the full set.
