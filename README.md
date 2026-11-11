# Mentora — AI tutor matching prototype

Mentora is a Flask web prototype for student and teacher profiles, explainable tutor recommendations, booking requests, in-site chat, demo classes, and a simulated checkout. Tutors come from teacher-created profiles; the project does not ship fictional Tutor A/B/C listings.

## Run locally

Requires Python 3.10 or newer.

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
4. Mentora filters for hard constraints and ranks eligible tutors with a transparent weighted baseline. If enough consented pair feedback is collected and a model is trained, recommendations can use logistic regression instead.
5. Students request regular or demo sessions. The teacher accepts or declines; accepted paid bookings proceed to a simulated checkout. No card details are collected and no real money is charged.
6. Students and teachers communicate in the booking chat. Teachers can send the video-call link there.

The default admin account is `admin@mentora.local` (set `MENTORA_ADMIN_EMAILS` to change it). This is a local educational prototype, not a production identity, payment, or video-conferencing service.

## Optional ML training

The model is trained on consented feedback about whether a student–tutor pair seemed like a good fit. Export the site's match feedback and train/evaluate using:

```bash
python export_ratings.py data/collected_ratings.csv
python train_model.py data/collected_ratings.csv
```

`data/ratings_template.csv` documents the columns and contains no fabricated responses. Training requires an adequate multi-class sample; report the actual sample size, respondent grouping, held-out metrics, and limitations. A trained model predicts the survey label, not learning outcomes. Until a model is available, the interface identifies the recommendation method as a rule-based baseline.

## Project files

- `app.py` — Flask routes, local database, authentication, profiles, bookings and chat
- `matcher.py` — eligibility filters, feature calculation, ranking and optional model loading
- `train_model.py` / `export_ratings.py` — grouped evaluation, training and feedback export
- `templates/` and `static/` — site pages and styling
