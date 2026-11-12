# Mentora model card — simulated-data prototype

- **Algorithm:** StandardScaler followed by Logistic Regression, fitted with scikit-learn 1.9.1.
- **Input:** nine numerical preference features from the shared `build_pair_features` function.
- **Output:** score for an artificial binary pair-fit label. This is not a validated probability of satisfaction or learning success.
- **Source:** 7,200 simulated pairs from 600 simulated students and a pool of 48 fictional tutor profiles. No survey responses or real person data were used.
- **Labels:** a Bernoulli draw from an invented preference utility with student-specific noise; see `generate_training_data.py`. This establishes a coding demonstration, not empirical preference research.
- **Split:** a fixed seed-42 75/25 split grouped by student, 5,400 training pairs and 1,800 test pairs; 450 training students and 150 test students. Tutors can appear in both sets.
- **Leakage controls:** student groups do not overlap; scaling is fitted on training rows only. No participant or tutor identifier is used as a feature. No threshold tuning is done on the test set.
- **Evaluation:** `evaluation.json` contains metrics of the held-out experiment. Ranking-related ROC-AUC is reported alongside threshold-dependent metrics and explicit baseline thresholds. Only one split is used; no confidence intervals or real-world generalization claims are supported.
- **Serving:** after evaluation is recorded, a final pipeline is fitted on all rows and saved to `matcher.joblib`. The reported test metrics describe the earlier evaluation pipeline. The app uses this final pipeline's predicted score directly, with no weighted-score blending.
- **Limitations:** artificial labels encode the generator's assumptions. Results should not be described as user-study findings. Coverage is limited to these languages, levels, subjects, and preference patterns. Real consented feedback, larger independent samples, new evaluation, and subgroup error checks are needed before claiming practical quality.
- **Reproduction:** `python generate_training_data.py` followed by `python train_model.py data/simulated_pairs.csv`. Install the pinned requirements using Python 3.12 or newer. The report records the dataset hash, dependency version, training timestamp and schema version.
- **Real feedback:** exported independently with consent and stable pseudonymous student IDs. Real and simulated label sources must not be mixed. Demo tutor listings remain excluded from human-feedback collection.

Only load this project's trusted model artifacts; Joblib can execute code during loading. Model artifacts, reports, and CSV files in this directory/repository contain simulation only.
