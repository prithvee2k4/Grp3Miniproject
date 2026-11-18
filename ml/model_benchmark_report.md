# Household electricity-bill model benchmark

## Result

The target is average monthly electricity bill in rupees, not electricity
consumption in kWh. The comparison used 7,575 training households and did not
use the previously inspected test split.

| Validation scheme | Model | MAE (₹) | RMSE (₹) | Pooled R² |
|---|---|---:|---:|---:|
| Random 5-fold | HGB, engineered | 287.64 | 433.89 | 0.573 |
| Random 5-fold | HGB baseline | 288.75 | 434.97 | 0.571 |
| Random 5-fold | XGBoost, engineered | 289.16 | 436.02 | 0.569 |
| Random 5-fold | CatBoost, engineered | 300.92 | 447.41 | 0.546 |
| State-held-out 5-fold | XGBoost, engineered | 458.84 | 620.48 | 0.127 |
| State-held-out 5-fold | HGB baseline | 462.96 | 626.08 | 0.111 |
| State-held-out 5-fold | CatBoost, engineered | 491.51 | 626.45 | 0.110 |
| State-held-out 5-fold | HGB, engineered | 468.72 | 633.05 | 0.091 |

Random-fold scores are pooled out-of-fold metrics; fold means and standard
deviations are in `output/ml_preprocessed/model_benchmark_summary.csv`.

## Interpretation

- The existing HistGradientBoosting model with engineered features remains the
  strongest candidate for predicting households drawn from the same mixture of
  states. Its improvement over baseline is small (about ₹1.10 MAE, or 0.38%),
  and it has the lowest fold MAE in only three of the five random folds.
- XGBoost is close but has higher MAE than engineered HGB in all five random
  folds. It has the best pooled scores when entire states are withheld, but
  state-held-out R² is still only 0.127.
- CatBoost did not improve either comparison with the tested fixed settings.
- All models have difficulty transferring to unseen states: pooled R² falls
  from roughly 0.55–0.57 on random folds to 0.09–0.13 on state-held-out folds.
  State-fold variability is substantial; one state-held-out fold has negative
  R² for every model. Read these scores as a warning about geographic shift,
  not a stable deployment guarantee.
- Error by bill range shows regression toward the mean. For HGB with engineered
  features, the lowest bill quintile is overpredicted by about ₹136 on average,
  while the highest quintile is underpredicted by about ₹508. CatBoost and
  XGBoost show the same pattern. See `model_benchmark_error_by_bill_range.csv`.

## Dashboard model and consumption-target decision

The Streamlit dashboard's editable household form uses a smaller, documented
set of profile inputs than the full benchmark. Its bill model has pooled
five-fold R² 0.511 on shuffled folds (MAE ₹319.79) and R² 0.187 on
state-held-out folds (MAE ₹445.22). These scores are specific to the
dashboard's shorter input form and are not directly comparable to the
full-feature benchmark above.

The available billed-units field, `q610_b_billed_units`, is observed for only
4,229 of 14,851 survey households (28.5%). Coverage varies sharply by state:
0.3% in Himachal Pradesh versus 93.8% in Andhra Pradesh and 89.9% in
Telangana. A pilot model has shuffled five-fold R² 0.236 (MAE 72.19 reported
units), but state-held-out R² is 0.007 (MAE 89.08). The codebook must still
confirm the units and billing period. The dashboard therefore reports target
coverage and validation only; it deliberately does not offer a billed-units
prediction.

The dashboard's bill profile model also reports global permutation importance
averaged across state-held-out folds. Treat it as predictive association, not
as a causal or household-specific explanation.

## Validation and limitations

- Both validation schemes operate only within the original training partition;
  the old test split is not used. That split has already been inspected and
  remains exploratory, not an untouched final estimate.
- Missingness/variance screening, imputers, and one-hot encoding are fit using
  each fold's training rows. The ten engineered features are deterministic
  survey-response rules and do not use the target.
- The random split measures interpolation across a similar state mix. The
  state-held-out split measures a harder unseen-state scenario; it is not a
  substitute for an external survey or a time-based validation.
- This is a fixed-setting benchmark, not a hyperparameter search. No model is
  established as universally best; the appropriate choice depends on whether
  deployment targets known-state households or new states.
- The survey weights are not used in training or metrics. Scores describe this
  observed sample, not a weighted national estimate.

## Reproduction

From the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe ml\benchmark_models.py --folds 5
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m flake8 ml\benchmark_models.py tests\test_model_benchmark.py
```

The script writes fold metrics, out-of-fold predictions, bill-range errors,
state-level errors, and validation metadata to `output/ml_preprocessed/`.
