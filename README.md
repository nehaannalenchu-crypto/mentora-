# Mentora — AI tutor matching prototype

Mentora is a Flask web prototype for student and teacher profiles, explainable tutor recommendations, booking requests, in-site chat, demo classes, and a simulated checkout. Teachers can create real profiles. For presentations, run `python seed_demo_tutors.py` to add eight clearly labeled fictional tutors with varied subjects, prices, and schedules; they have no real teacher behind them and no fabricated reviews.

## Run locally

Requires Python 3.12 or newer. ML dependency versions are pinned to match the bundled trained model.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5001>. The local SQLite database is created under `instance/` and excluded from Git. For a Google Cloud Run classroom demo, the `Procfile` configures Gunicorn; the SQLite database remains temporary and is not suitable for persistent multi-user hosting.

## Main flow

1. Students and teachers create accounts using email plus a one-time code. Without SMTP settings, the page displays a demo code; configure `MENTORA_SMTP_HOST`, `MENTORA_SMTP_PORT`, `MENTORA_SMTP_USERNAME`, `MENTORA_SMTP_PASSWORD`, and `MENTORA_FROM_EMAIL` to send actual email.
2. Students enter subjects, study level, budget, tutor gender preference, availability, languages, learning style, class format, delivery mode, session length, and goal.
3. Teachers add education, subjects, grades, languages, rate, availability, teaching approach, experience, bio, and optional demo-class details. Profiles remain hidden from recommendations until approved in the admin queue.
4. Mentora ranks approved tutors by closeness across subjects, schedule, language, learning style, level, budget, teacher gender preference, class format, and delivery mode. Imperfect matches remain visible with the differences explained. The bundled Logistic Regression model ranks these profiles using nine shared features. Its current training data is simulated and the website labels this explicitly. A weighted preference baseline remains available if a compatible model cannot be loaded.
5. Students request regular or demo sessions. The teacher accepts or declines; accepted paid bookings proceed to a simulated checkout. No card details are collected and no real money is charged.
6. Students and teachers communicate in the booking chat. Teachers can send the video-call link there.

The default admin account is `admin@mentora.local` (set `MENTORA_ADMIN_EMAILS` to change it). This is a local educational prototype, not a production identity, payment, or video-conferencing service.

## Actual ML prototype

The app ships a real fitted `StandardScaler → LogisticRegression` pipeline at `models/matcher.joblib`. It is trained on **simulated data**, not on real students or tutoring outcomes. Names, email addresses, and participant IDs are not features. Visit `/matching-method` for its data source, held-out evaluation, baselines, and limitations.

Reproduce the dataset and training:

```bash
python generate_training_data.py
python train_model.py data/simulated_pairs.csv
```

The generator uses seed 42, 600 simulated students, 48 fictional tutor profiles, and 12 distinct pairs per student: 7,200 labeled pairs. Labels are random samples from an explicitly invented preference utility, with student-specific noise; the coefficients and process are documented in `generate_training_data.py`. These are not fabricated survey responses. `data/simulated_pairs.metadata.json` records provenance.

Nine features cover subject overlap, schedule overlap, language, teaching style, level, budget, optional teacher gender preference, class format, and delivery mode. Training and serving use the same extractor. Partial matches stay visible. With a trained model, its `predict_proba` score directly determines ranking; it is not blended with the baseline.

Evaluation uses a fixed 75/25 split **grouped by simulated student**: 5,400 training pairs and 1,800 held-out pairs, with no student appearing on both sides. Tutors may appear in both sets. Scaling is fitted only on training rows. `models/evaluation.json` records accuracy, precision, recall, F1, ROC-AUC, Brier score, a confusion matrix, and comparisons against weighted and majority-label baselines. Thresholds are fixed before evaluation, not tuned on the held-out set. After evaluation, the serving model is refitted on all rows. Evaluation metrics refer to the held-out experiment, not to this final refitted model.

**Interpretation:** these results only measure recovery of the invented simulated pattern. They do not validate real tutor quality, student satisfaction, or learning improvement. The UI presents this as a simulated-data ML prototype. The next research step is consented human feedback and fresh evaluation before making real performance claims.

### Replace simulation with real feedback later

For approved real teacher profiles, students can explicitly consent to using a 1–5 pair-fit rating for training. Ratings 1–2 map to 0, 4–5 to 1, and 3 is excluded. Fictional demo listings cannot supply these training ratings. A random participant ID is stored per student account and stays stable across logins so evaluation can keep each student's rows together. This makes the exported data pseudonymous, not fully anonymous. Account-to-participant mappings remain in the local database and are not exported.

```bash
python export_ratings.py data/collected_ratings.csv
python train_model.py data/collected_ratings.csv
```

The exporter uses the app's `instance/mentora.sqlite3` database (or `MENTORA_DATABASE` / `--database` override), exports only consented labels, and keeps the latest rating per student–tutor pair. The CSV must declare `label_source=human_feedback`; simulated and human labels cannot be mixed. At least 20 pairs from 5 students are required to attempt a split, but that minimum does not establish reliability. Collect substantially more independent feedback and report uncertainty. Training refuses invalid/missing values and incompatible old feature schemas.

To switch back to the simulation, rerun the first two commands. Model provenance follows the artifact. Missing, corrupt, incompatible-schema, or version-mismatched artifacts fall back to the weighted baseline instead of being silently used. Only load trusted model files; Joblib is not a safe interchange format for untrusted uploads.

### Checks

```bash
python -m unittest discover -s tests -v
```

## Project files

- `app.py` — Flask routes, local database, authentication, profiles, bookings and chat
- `matcher.py` — shared feature calculation, weighted fallback and model loading/inference
- `generate_training_data.py` — reproducible simulated profiles and labeled pairs
- `train_model.py` / `export_ratings.py` — grouped evaluation, training and consented feedback export
- `models/` — the bundled simulated-data model, evaluation report and model card
- `templates/` and `static/` — site pages and styling

Demo tutor recommendation feedback is hidden so fictional listings are not presented as real training evidence. The separately generated simulation is explicitly labeled.
