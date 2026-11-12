**Repository:** https://github.com/Soham-Paranjpe27/CapStoneproj

# NoShowGuard: predicting and explaining missed clinic appointments

> **Before submitting, finish the three items marked TODO below.** They need a run on your own machine (Colab and MySQL Workbench), so they could not be filled in ahead of time. Delete this note afterwards.

## Summary (one finding per Part)

1. **Data (Part 1):** after removing 20 duplicate rows and 10 rows with no usable date or status, 990 clean appointments remain, of which 218 (22.0%) were no-shows.
2. **SQL (Part 2):** no-shows climb sharply with lead time, from 16.2% for bookings made 1-7 days ahead to 60.5% for bookings made 31 or more days ahead.
3. **Machine learning (Part 3):** logistic regression was chosen (cross-validated ROC-AUC 0.774 on the training set, 0.789 on the test set); at a threshold of 0.35 it flags 49 of 198 test bookings and catches 57% of the real no-shows (28 of 49).
4. **GenAI (Part 4):** TODO after your Gemini run: state how many of the 9 non-blank notes got a valid, expected root cause, and how many of the notes classed `forgot_or_no_reminder` belonged to appointments with `reminder_sent = No` (expected 2 of 2 if N001 and N007 are classified as in the key; 5 of the 10 notes belong to appointments without a reminder).

## Recommendation

The front desk should phone, the evening before, every booking whose risk score is 0.35 or higher, and reminders should go out for every appointment by default. The overall no-show rate is 22.0% (218 of 990), but it is not evenly spread: bookings made more than two weeks ahead are the main risk (30.0% for 15-30 days and 60.5% for 31+ days, against 16.2% for 1-7 days), and appointments without a reminder were missed about twice as often as those with one (30.8% against 15.3%). In the fitted model, holding the other features constant, a reminder is associated with roughly 69% lower odds of a no-show (odds multiplier 0.31). On the held-out test set the 0.35 threshold means 49 calls for 198 bookings, of which 28 reach a patient who would otherwise not have come, so about 57% of the no-shows are caught for a call list of a quarter of the bookings. Clinic differences should be read with care: Gomti Nagar Care Point has the highest rate (33.3%) but only 90 appointments, while among the larger clinics Hadapsar Care Centre (27.3% of 143) and Malviya Nagar Clinic (26.8% of 142) are the ones to start with. The follow-up notes point the same way where they can: TODO after your Gemini run, add one sentence with your crosstab result.

## Model chosen and why

The model was chosen from training data only, by 5-fold cross-validated ROC-AUC:

| Model | CV ROC-AUC (training set) |
|---|---|
| Logistic regression (scaler inside a pipeline) | 0.774 |
| Tuned random forest (`max_depth=3`, `n_estimators=200`) | 0.767 |
| Random forest (100 trees) | 0.747 |
| Decision tree (depth 3) | 0.678 |

Logistic regression had the highest score, so it was the final model. The threshold was also chosen without the test set, from out-of-fold training probabilities: 0.35 gave an F1 of 0.489 (precision 0.513, recall 0.467, 154 flagged) against 0.347 at 0.5 (precision 0.575, recall 0.249, 73 flagged). Recall matters more here because a missed warning costs an empty slot while an extra call costs about a minute. The single test-set evaluation at 0.35 gave precision 0.571, recall 0.571, 49 flagged and ROC-AUC 0.789. The baseline of always predicting "Attended" is 0.7525 accurate, and the unlimited decision tree scored below it on the test set (0.672, with 1.00 on the training set), which is why accuracy alone was not used to judge the models.

## Limitations

1. **The 10 notes are an illustration.** They cannot show the real mix of reasons across all clinics.
2. **The data are observational.** The reminder result is an association in this model, not proof that reminders cause fewer no-shows.
3. **The evaluation is optimistic in small ways.** Medians and outlier fences were computed on all 990 rows before the split, the same patient can appear in both train and test, and the split is random rather than past-to-future, on a single train/test split.

In practice the threshold would be set by how many calls the front desk can make in an evening, against the cost of an empty slot.

## SQL notes (Part 2)

- Loaded 8 clinics and 990 appointments; `SUM(no_show)` is 218, the same as in Python.
- Duplicate insert of `A0001`: TODO paste the exact error message from MySQL Workbench here (it is a duplicate-key error on the primary key).
- Without the `HAVING` line, Gomti Nagar Care Point (Lucknow) has the highest rate, 33.3%, on only 90 appointments. The pandas `groupby` counts match the SQL counts exactly.
- Overall no-show rate used in Task 2.5: 22.02%.

## Files

| File | What it is |
|---|---|
| `noshowguard_capstone.ipynb` | All Python for Parts 1-4 |
| `queries.sql` | Table creation and the Part 2 queries |
| `appointments.csv`, `clinics.csv`, `followup_notes.csv` | Input data |
| `appointments_clean.csv` | Cleaned table, 990 rows and 16 columns |
| `rejected_rows.json` | The 32 rows rejected by the csv-module type check |
| `test_risk_scores.csv` | Risk score and flag for the 198 test rows |
| `clinics_enriched.csv`, `load_data.sql`, `notes_analysed.csv` | Written when the notebook is run (they need the weather API and the Gemini key) |

## How to run

1. Open the notebook in Colab and upload the three input CSVs and `queries.sql` to the session.
2. Run all cells. The Gemini key is asked for when the Part 4 cell runs and is not stored anywhere in this repository.
3. In MySQL Workbench, run `queries.sql` up to the table creation, then `load_data.sql`, then the rest of `queries.sql`.

Reference versions: scikit-learn 1.5.0, pandas 2.3, NumPy 1.26.
