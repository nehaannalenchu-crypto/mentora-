"""Mentora's local tutoring marketplace prototype."""

from __future__ import annotations

import json
import os
import secrets
import smtplib
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from email.message import EmailMessage
from functools import wraps
from pathlib import Path
from uuid import uuid4
from typing import Iterator

from flask import (
    Flask, abort, flash, g, redirect, render_template, request, session,
    url_for,
)

from matcher import load_trained_model, rank_tutors


ROOT = Path(__file__).parent
DATABASE_PATH = Path(os.environ.get("MENTORA_DATABASE", ROOT / "instance" / "mentora.sqlite3"))
SUBJECTS = [
    "Computer Science / Programming", "AI & Machine Learning", "Mathematics",
    "Physics", "Electronics", "Electrical Engineering", "English", "French",
    "Other",
]
LEVELS = ["School (K–12)", "Undergraduate", "Postgraduate", "Competitive exams", "Other"]
LANGUAGES = ["English", "French", "Creole", "Hindi", "Malayalam", "Tamil", "Telugu", "Kannada", "Other"]
AVAILABILITY = [
    "Monday morning", "Monday afternoon", "Monday evening",
    "Tuesday morning", "Tuesday afternoon", "Tuesday evening",
    "Wednesday morning", "Wednesday afternoon", "Wednesday evening",
    "Thursday morning", "Thursday afternoon", "Thursday evening",
    "Friday morning", "Friday afternoon", "Friday evening",
    "Saturday morning", "Saturday afternoon", "Saturday evening",
    "Sunday morning", "Sunday afternoon", "Sunday evening",
]
LEARNING_STYLES = ["Step-by-step explanations", "Practice problems", "Visual explanations", "Discussion and questions"]
CLASS_FORMATS = ["1-on-1", "Group class"]
DELIVERY_MODES = ["Online", "In person"]
ADMIN_EMAILS = {
    item.strip().lower()
    for item in os.environ.get("MENTORA_ADMIN_EMAILS", "admin@mentora.local").split(",")
    if item.strip()
}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db() -> None:
    with db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL CHECK(role IN ('student','teacher','admin')),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS profiles (
                user_id INTEGER PRIMARY KEY REFERENCES users(id),
                profile_json TEXT NOT NULL,
                complete INTEGER NOT NULL DEFAULT 0,
                verification_status TEXT NOT NULL DEFAULT 'not_applicable',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL REFERENCES users(id),
                teacher_id INTEGER NOT NULL REFERENCES users(id),
                session_type TEXT NOT NULL CHECK(session_type IN ('demo','regular')),
                session_date TEXT NOT NULL,
                time_slot TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                amount REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'requested',
                payment_status TEXT NOT NULL DEFAULT 'not_required',
                requested_date TEXT,
                requested_slot TEXT,
                reschedule_previous_status TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_id INTEGER NOT NULL REFERENCES bookings(id),
                sender_id INTEGER NOT NULL REFERENCES users(id),
                body TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'text',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS saved_tutors (
                student_id INTEGER NOT NULL REFERENCES users(id),
                teacher_id INTEGER NOT NULL REFERENCES users(id),
                created_at TEXT NOT NULL,
                PRIMARY KEY(student_id, teacher_id)
            );
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                is_read INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_id INTEGER NOT NULL UNIQUE REFERENCES bookings(id),
                student_id INTEGER NOT NULL REFERENCES users(id),
                teacher_id INTEGER NOT NULL REFERENCES users(id),
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                comment TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reporter_id INTEGER NOT NULL REFERENCES users(id),
                teacher_id INTEGER NOT NULL REFERENCES users(id),
                reason TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS blocks (
                blocker_id INTEGER NOT NULL REFERENCES users(id),
                blocked_id INTEGER NOT NULL REFERENCES users(id),
                created_at TEXT NOT NULL,
                PRIMARY KEY(blocker_id, blocked_id)
            );
            CREATE TABLE IF NOT EXISTS match_respondents (
                user_id INTEGER PRIMARY KEY REFERENCES users(id),
                respondent_id TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS match_requests (
                request_id TEXT PRIMARY KEY,
                respondent_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS shown_matches (
                request_id TEXT NOT NULL,
                respondent_id TEXT NOT NULL,
                tutor_id TEXT NOT NULL,
                feature_json TEXT NOT NULL,
                PRIMARY KEY (request_id, tutor_id)
            );
            CREATE TABLE IF NOT EXISTS feedback (
                request_id TEXT NOT NULL,
                respondent_id TEXT NOT NULL,
                tutor_id TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                good_fit INTEGER,
                training_consent INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (request_id, tutor_id)
            );
            """
        )
        feedback_columns = {row[1] for row in connection.execute("PRAGMA table_info(feedback)")}
        if "training_consent" not in feedback_columns:
            connection.execute("ALTER TABLE feedback ADD COLUMN training_consent INTEGER NOT NULL DEFAULT 0")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_profile(user_id: int) -> dict:
    with db() as connection:
        row = connection.execute("SELECT profile_json FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    return json.loads(row["profile_json"]) if row else {}


def get_respondent_id(user_id: int) -> str:
    # Stable random ID prevents one student leaking across grouped data splits.
    with db() as connection:
        connection.execute("INSERT OR IGNORE INTO match_respondents(user_id,respondent_id) VALUES(?,?)", (user_id, uuid4().hex))
        return connection.execute("SELECT respondent_id FROM match_respondents WHERE user_id=?", (user_id,)).fetchone()[0]


def notify(user_id: int, title: str, body: str) -> None:
    with db() as connection:
        connection.execute(
            "INSERT INTO notifications(user_id,title,body,created_at) VALUES(?,?,?,?)",
            (user_id, title, body, now_iso()),
        )


def require_login(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Sign in to continue.", "info")
            return redirect(url_for("home"))
        with db() as connection:
            user = connection.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()
        if not user:
            session.clear()
            return redirect(url_for("home"))
        g.user = dict(user)
        return view(*args, **kwargs)
    return wrapped


def require_role(*roles):
    def decorate(view):
        @wraps(view)
        @require_login
        def wrapped(*args, **kwargs):
            if g.user["role"] not in roles:
                abort(403)
            return view(*args, **kwargs)
        return wrapped
    return decorate


def email_otp(recipient: str, code: str) -> bool:
    host = os.environ.get("MENTORA_SMTP_HOST")
    username = os.environ.get("MENTORA_SMTP_USERNAME")
    password = os.environ.get("MENTORA_SMTP_PASSWORD")
    if not (host and username and password):
        return False
    message = EmailMessage()
    message["Subject"] = "Your Mentora sign-in code"
    message["From"] = os.environ.get("MENTORA_FROM_EMAIL", username)
    message["To"] = recipient
    message.set_content(f"Your Mentora one-time sign-in code is {code}. It expires in 10 minutes.")
    with smtplib.SMTP(host, int(os.environ.get("MENTORA_SMTP_PORT", "587")), timeout=10) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(message)
    return True


def form_list(name: str) -> list[str]:
    return [value for value in request.form.getlist(name) if value]


def save_profile(user_id: int, values: dict, *, complete: bool = True, verification: str = "not_applicable") -> None:
    with db() as connection:
        connection.execute(
            """INSERT INTO profiles(user_id,profile_json,complete,verification_status,updated_at)
               VALUES(?,?,?,?,?)
               ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json,
               complete=excluded.complete, verification_status=excluded.verification_status,
               updated_at=excluded.updated_at""",
            (user_id, json.dumps(values, ensure_ascii=False), int(complete), verification, now_iso()),
        )


@app.context_processor
def shared_page_data():
    current = None
    unread = 0
    if session.get("user_id"):
        with db() as connection:
            row = connection.execute("SELECT id,email,role FROM users WHERE id=?", (session["user_id"],)).fetchone()
            if row:
                current = dict(row)
                unread = connection.execute("SELECT COUNT(*) FROM notifications WHERE user_id=? AND is_read=0", (row["id"],)).fetchone()[0]
    model_bundle = load_trained_model()
    return {
        "current_user": current,
        "notification_count": unread,
        "model_ready": model_bundle is not None,
        "model_source": model_bundle.get("training_source") if model_bundle else None,
        "demo_mode": not all(os.environ.get(name) for name in ("MENTORA_SMTP_HOST", "MENTORA_SMTP_USERNAME", "MENTORA_SMTP_PASSWORD")),
    }


@app.get("/")
def home():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template(
        "auth.html",
        suggested_role=request.args.get("role", "student"),
        pending_email=session.get("pending_email"),
        demo_code=session.get("demo_otp"),
    )


@app.post("/auth/start")
def auth_start():
    email = request.form.get("email", "").strip().lower()
    requested_role = request.form.get("role", "student")
    if "@" not in email or "." not in email.split("@")[-1]:
        flash("Enter a valid email address.", "error")
        return redirect(url_for("home"))
    if requested_role not in {"student", "teacher"}:
        flash("Choose Student or Teacher.", "error")
        return redirect(url_for("home"))
    role = "admin" if email in ADMIN_EMAILS else requested_role
    code = f"{secrets.randbelow(1_000_000):06d}"
    session["pending_email"] = email
    session["pending_role"] = role
    session["otp_code"] = code
    session["otp_expires"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()
    session["otp_attempts"] = 0
    try:
        sent = email_otp(email, code)
    except (OSError, smtplib.SMTPException) as error:
        app.logger.warning("OTP email delivery failed: %s", error)
        flash("Email delivery failed. Check the mail settings or use local demo mode.", "error")
        session.pop("otp_code", None)
        return redirect(url_for("home"))
    if sent:
        session.pop("demo_otp", None)
        flash("A one-time code was sent to your email.", "success")
    else:
        session["demo_otp"] = code
        flash("Local demo mode: use the one-time code shown below. No email was sent.", "info")
    return redirect(url_for("home"))


@app.post("/auth/verify")
def auth_verify():
    submitted = request.form.get("otp", "").strip()
    expires = session.get("otp_expires", 0)
    attempts = session.get("otp_attempts", 0)
    if not session.get("otp_code") or datetime.now(timezone.utc).timestamp() > expires:
        session.pop("otp_code", None)
        flash("That code expired. Request a new one.", "error")
        return redirect(url_for("home"))
    if attempts >= 5:
        session.pop("otp_code", None)
        flash("Too many attempts. Request a new code.", "error")
        return redirect(url_for("home"))
    if not secrets.compare_digest(submitted, session["otp_code"]):
        session["otp_attempts"] = attempts + 1
        flash("That code did not match. Try again.", "error")
        return redirect(url_for("home"))
    email = session["pending_email"]
    requested_role = session["pending_role"]
    with db() as connection:
        user = connection.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user:
            user_id = user["id"]
            role = user["role"]
        else:
            cursor = connection.execute(
                "INSERT INTO users(email,role,created_at) VALUES(?,?,?)",
                (email, requested_role, now_iso()),
            )
            user_id = cursor.lastrowid
            role = requested_role
    session.clear()
    session["user_id"] = user_id
    session["respondent_id"] = get_respondent_id(user_id)
    if role == "admin":
        return redirect(url_for("admin_dashboard"))
    with db() as connection:
        profile = connection.execute("SELECT complete FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    if not profile or not profile["complete"]:
        return redirect(url_for("profile_setup"))
    return redirect(url_for("dashboard"))


@app.post("/auth/resend")
def auth_resend():
    if not session.get("pending_email"):
        return redirect(url_for("home"))
    return auth_start_from_pending()


def auth_start_from_pending():
    email = session["pending_email"]
    code = f"{secrets.randbelow(1_000_000):06d}"
    session["otp_code"] = code
    session["otp_expires"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).timestamp()
    session["otp_attempts"] = 0
    if email_otp(email, code):
        session.pop("demo_otp", None)
        flash("A new code was sent.", "success")
    else:
        session["demo_otp"] = code
        flash("Local demo mode: use the new code shown below. No email was sent.", "info")
    return redirect(url_for("home"))


@app.get("/logout")
def logout():
    session.clear()
    flash("You have signed out.", "success")
    return redirect(url_for("home"))


@app.get("/profile")
@require_login
def profile_setup():
    if g.user["role"] == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("student_profile" if g.user["role"] == "student" else "teacher_profile"))


@app.route("/profile/student", methods=["GET", "POST"])
@require_role("student")
def student_profile():
    existing = get_profile(g.user["id"])
    if request.method == "POST":
        subjects = form_list("subjects")
        availability = form_list("availability")
        languages = form_list("languages")
        formats = form_list("formats")
        study_level = request.form.get("study_level", "")
        gender = request.form.get("teacher_gender", "any")
        try:
            budget = float(request.form.get("budget_max", ""))
            if budget <= 0:
                raise ValueError
        except ValueError:
            flash("Enter a budget greater than zero.", "error")
            return redirect(url_for("student_profile"))
        if not subjects or any(value not in SUBJECTS for value in subjects):
            flash("Choose at least one subject.", "error")
            return redirect(url_for("student_profile"))
        if not availability or any(value not in AVAILABILITY for value in availability):
            flash("Choose at least one available time.", "error")
            return redirect(url_for("student_profile"))
        if any(value not in LANGUAGES for value in languages) or any(value not in CLASS_FORMATS for value in formats):
            flash("Choose valid language and class-format preferences.", "error")
            return redirect(url_for("student_profile"))
        if gender not in {"female", "male", "any"}:
            flash("Choose a teacher preference.", "error")
            return redirect(url_for("student_profile"))
        language_other = request.form.get("language_other", "").strip()
        subject_other = request.form.get("subject_other", "").strip()
        if "Other" in subjects and subject_other:
            subjects.append(subject_other)
        if "Other" in languages and language_other:
            languages.append(language_other)
        values = {
            "name": request.form.get("name", "").strip(),
            "subjects": subjects,
            "subject_other": subject_other,
            "study_level": study_level,
            "budget_max": budget,
            "teacher_gender": gender,
            "availability": availability,
            "languages": languages,
            "language_other": language_other,
            "formats": formats,
            "delivery_modes": form_list("delivery_modes"),
            "learning_styles": form_list("learning_styles"),
            "session_minutes": request.form.get("session_minutes", "60"),
            "goal": request.form.get("goal", "").strip(),
            "notes": request.form.get("notes", "").strip(),
        }
        save_profile(g.user["id"], values)
        flash("Student preferences saved. Mentora is ready to find matching tutors.", "success")
        return redirect(url_for("tutor_matches"))
    return render_template(
        "student_profile.html", profile=existing, subjects=SUBJECTS, levels=LEVELS,
        languages=LANGUAGES, availability=AVAILABILITY, formats=CLASS_FORMATS,
        modes=DELIVERY_MODES, styles=LEARNING_STYLES,
    )


@app.route("/profile/teacher", methods=["GET", "POST"])
@require_role("teacher")
def teacher_profile():
    existing = get_profile(g.user["id"])
    verification = "pending"
    with db() as connection:
        status_row = connection.execute("SELECT verification_status FROM profiles WHERE user_id=?", (g.user["id"],)).fetchone()
        if status_row:
            verification = status_row["verification_status"]
    if request.method == "POST":
        subjects = form_list("subjects")
        grades = form_list("grades")
        availability = form_list("availability")
        languages = form_list("languages")
        modes = form_list("delivery_modes")
        formats = form_list("formats")
        name = request.form.get("display_name", "").strip()
        qualification = request.form.get("qualification", "").strip()
        try:
            rate = float(request.form.get("hourly_rate", ""))
            demo_price = float(request.form.get("demo_price", "0") or "0")
            demo_minutes = int(request.form.get("demo_minutes", "30"))
            experience = max(0, int(request.form.get("experience_years", "0") or "0"))
            if rate <= 0 or demo_price < 0 or demo_minutes not in {15, 20, 30, 45, 60}:
                raise ValueError
        except ValueError:
            flash("Check the hourly rate, demo price, duration, and experience fields.", "error")
            return redirect(url_for("teacher_profile"))
        if not name or not qualification or not subjects or not grades or not availability or not languages:
            flash("Complete your name, qualification, subjects, levels, languages, and availability.", "error")
            return redirect(url_for("teacher_profile"))
        if any(value not in SUBJECTS for value in subjects) or any(value not in LEVELS for value in grades):
            flash("Choose subjects and levels from the provided options.", "error")
            return redirect(url_for("teacher_profile"))
        if any(value not in AVAILABILITY for value in availability) or any(value not in LANGUAGES for value in languages):
            flash("Choose valid language and availability options.", "error")
            return redirect(url_for("teacher_profile"))
        if any(value not in CLASS_FORMATS for value in formats) or any(value not in DELIVERY_MODES for value in modes):
            flash("Choose valid class formats and delivery modes.", "error")
            return redirect(url_for("teacher_profile"))
        language_other = request.form.get("language_other", "").strip()
        subject_other = request.form.get("subject_other", "").strip()
        if "Other" in subjects and subject_other:
            subjects.append(subject_other)
        if "Other" in languages and language_other:
            languages.append(language_other)
        values = {
            "display_name": name,
            "gender": request.form.get("gender", "").strip().lower(),
            "qualification": qualification,
            "certificate_details": request.form.get("certificate_details", "").strip(),
            "subjects": subjects,
            "subject_other": subject_other,
            "grades": grades,
            "languages": languages,
            "language_other": language_other,
            "hourly_rate": rate,
            "availability": availability,
            "delivery_modes": modes,
            "formats": formats,
            "teaching_style": request.form.get("teaching_style", "").strip(),
            "experience_years": experience,
            "bio": request.form.get("bio", "").strip(),
            "photo_url": request.form.get("photo_url", "").strip(),
            "offers_demo": request.form.get("offers_demo") == "on",
            "demo_price": demo_price,
            "demo_minutes": demo_minutes,
        }
        save_profile(g.user["id"], values, verification="pending")
        flash("Teacher profile saved. It will appear in recommendations after admin review.", "success")
        return redirect(url_for("dashboard"))
    return render_template(
        "teacher_profile.html", profile=existing, verification=verification,
        subjects=SUBJECTS, levels=LEVELS, languages=LANGUAGES,
        availability=AVAILABILITY, formats=CLASS_FORMATS, modes=DELIVERY_MODES,
        styles=LEARNING_STYLES,
    )


@app.get("/dashboard")
@require_login
def dashboard():
    if g.user["role"] == "admin":
        return redirect(url_for("admin_dashboard"))
    role = g.user["role"]
    profile = get_profile(g.user["id"])
    verification_status = "not_applicable"
    with db() as connection:
        if role == "student":
            rows = connection.execute(
                """SELECT b.*, u.email AS teacher_email, p.profile_json AS teacher_profile
                   FROM bookings b JOIN users u ON u.id=b.teacher_id
                   LEFT JOIN profiles p ON p.user_id=u.id
                   WHERE b.student_id=? ORDER BY b.created_at DESC""", (g.user["id"],)
            ).fetchall()
            saved = connection.execute(
                """SELECT u.id,p.profile_json,p.verification_status FROM saved_tutors s
                   JOIN users u ON u.id=s.teacher_id JOIN profiles p ON p.user_id=u.id
                   WHERE s.student_id=? ORDER BY s.created_at DESC""", (g.user["id"],)
            ).fetchall()
        else:
            rows = connection.execute(
                """SELECT b.*, u.email AS student_email, p.profile_json AS student_profile
                   FROM bookings b JOIN users u ON u.id=b.student_id
                   LEFT JOIN profiles p ON p.user_id=u.id
                   WHERE b.teacher_id=? ORDER BY b.created_at DESC""", (g.user["id"],)
            ).fetchall()
            saved = []
        notices = connection.execute(
            "SELECT * FROM notifications WHERE user_id=? ORDER BY created_at DESC LIMIT 12", (g.user["id"],)
        ).fetchall()
        if role == "teacher":
            status_row = connection.execute("SELECT verification_status FROM profiles WHERE user_id=?", (g.user["id"],)).fetchone()
            if status_row:
                verification_status = status_row["verification_status"]
    bookings = []
    for row in rows:
        item = dict(row)
        if role == "student":
            item["counterpart"] = json.loads(item.pop("teacher_profile") or "{}")
            item["counterpart_name"] = item["counterpart"].get("display_name", "Tutor")
        else:
            item["counterpart"] = json.loads(item.pop("student_profile") or "{}")
            item["counterpart_name"] = item["counterpart"].get("name") or item["student_email"]
        bookings.append(item)
    saved_tutors = []
    for row in saved:
        item = dict(row)
        item["profile"] = json.loads(item.pop("profile_json"))
        item["id"] = item.pop("id")
        saved_tutors.append(item)
    return render_template(
        "dashboard.html", profile=profile, bookings=bookings, saved_tutors=saved_tutors,
        notifications=[dict(note) for note in notices], verification_status=verification_status,
        availability=AVAILABILITY, today=date.today().isoformat(),
    )


@app.post("/notifications/read")
@require_login
def notifications_read():
    with db() as connection:
        connection.execute("UPDATE notifications SET is_read=1 WHERE user_id=?", (g.user["id"],))
    return redirect(url_for("dashboard"))


@app.get("/tutors")
@require_role("student")
def tutor_matches():
    student = get_profile(g.user["id"])
    if not student:
        return redirect(url_for("student_profile"))
    query = request.args.get("q", "").strip()
    sort = request.args.get("sort", "recommended")
    with db() as connection:
        rows = connection.execute(
            """SELECT u.id,p.profile_json,p.verification_status FROM users u
               JOIN profiles p ON p.user_id=u.id WHERE u.role='teacher' AND p.complete=1
               AND p.verification_status='approved' ORDER BY u.id DESC"""
        ).fetchall()
        saved_ids = {
            row[0] for row in connection.execute("SELECT teacher_id FROM saved_tutors WHERE student_id=?", (g.user["id"],)).fetchall()
        }
        blocked_ids = {
            row[0] for row in connection.execute("SELECT blocked_id FROM blocks WHERE blocker_id=?", (g.user["id"],)).fetchall()
        }
        review_rows = connection.execute("SELECT teacher_id,AVG(rating) AS average,COUNT(*) AS total FROM reviews GROUP BY teacher_id").fetchall()
    review_stats = {row["teacher_id"]: (float(row["average"]), int(row["total"])) for row in review_rows}
    tutors = []
    for row in rows:
        if row["id"] in blocked_ids:
            continue
        tutor = json.loads(row["profile_json"])
        tutor.update({"id": str(row["id"]), "verification_status": row["verification_status"], "user_id": row["id"]})
        average, total = review_stats.get(row["id"], (0.0, 0))
        tutor["rating"] = average
        tutor["review_count"] = total
        tutors.append(tutor)
    ranked = rank_tutors(student, tutors, load_trained_model())
    if query:
        query_lower = query.casefold()
        ranked = [item for item in ranked if query_lower in " ".join([
            item.get("display_name", ""), item.get("qualification", ""), item.get("bio", ""),
            *item.get("subjects", []), *item.get("grades", []), *item.get("languages", []),
        ]).casefold()]
    if sort == "rating":
        ranked.sort(key=lambda item: (item.get("rating", 0), item["score"]), reverse=True)
    elif sort == "price":
        ranked.sort(key=lambda item: item.get("hourly_rate", 0))
    for item in ranked:
        item["saved"] = int(item["id"]) in saved_ids
    request_id = None
    if ranked:
        respondent_id = get_respondent_id(g.user["id"])
        session["respondent_id"] = respondent_id
        request_id = uuid4().hex
        timestamp = now_iso()
        with db() as connection:
            connection.execute("INSERT INTO match_requests(request_id,respondent_id,created_at) VALUES(?,?,?)", (request_id, respondent_id, timestamp))
            connection.executemany(
                "INSERT INTO shown_matches(request_id,respondent_id,tutor_id,feature_json) VALUES(?,?,?,?)",
                [(request_id, respondent_id, item["id"], json.dumps(item["features"])) for item in ranked],
            )
    return render_template(
        "matches.html", student=student, matches=ranked, request_id=request_id,
        has_pending_tutors=has_pending_tutors(), query=query, sort=sort,
    )


def has_pending_tutors() -> bool:
    with db() as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM profiles p JOIN users u ON u.id=p.user_id WHERE u.role='teacher' AND p.verification_status='pending' AND p.complete=1"
        ).fetchone()[0] > 0


@app.get("/tutors/<int:teacher_id>")
@require_role("student")
def tutor_detail(teacher_id: int):
    with db() as connection:
        row = connection.execute(
            """SELECT u.id,p.profile_json,p.verification_status FROM users u JOIN profiles p ON p.user_id=u.id
               WHERE u.id=? AND u.role='teacher' AND p.complete=1 AND p.verification_status='approved'""",
            (teacher_id,),
        ).fetchone()
    if not row:
        abort(404)
    tutor = json.loads(row["profile_json"])
    tutor["id"] = teacher_id
    return render_template("tutor_detail.html", tutor=tutor, today=date.today().isoformat())


@app.post("/feedback")
@require_role("student")
def feedback():
    if request.form.get("training_consent") != "1":
        flash("Please agree before using a rating to improve matching.", "info")
        return redirect(url_for("tutor_matches"))
    request_id = request.form.get("request_id", "")
    tutor_id = request.form.get("tutor_id", "")
    try:
        rating = int(request.form.get("rating", ""))
    except ValueError:
        abort(400, "Choose a rating from 1 to 5.")
    if rating not in range(1, 6):
        abort(400, "Choose a rating from 1 to 5.")
    respondent_id = session.get("respondent_id")
    with db() as connection:
        profile_row = connection.execute("SELECT profile_json FROM profiles WHERE user_id=?", (tutor_id,)).fetchone()
        if profile_row and json.loads(profile_row["profile_json"]).get("is_demo"):
            abort(400, "Demo profiles are excluded from training feedback.")
        shown = connection.execute(
            "SELECT 1 FROM shown_matches WHERE request_id=? AND tutor_id=? AND respondent_id=?",
            (request_id, tutor_id, respondent_id),
        ).fetchone()
        if not shown:
            abort(400, "That tutor was not part of this match list.")
        good_fit = 1 if rating >= 4 else 0 if rating <= 2 else None
        connection.execute(
            "INSERT OR REPLACE INTO feedback(request_id,respondent_id,tutor_id,rating,good_fit,training_consent,created_at) VALUES(?,?,?,?,?,?,?)",
            (request_id, respondent_id, tutor_id, rating, good_fit, 1, now_iso()),
        )
    flash("Thanks for rating the recommendation.", "success")
    return redirect(url_for("tutor_matches"))


@app.post("/tutors/<int:teacher_id>/save")
@require_role("student")
def save_tutor(teacher_id: int):
    with db() as connection:
        exists = connection.execute("SELECT 1 FROM users WHERE id=? AND role='teacher'", (teacher_id,)).fetchone()
        if not exists:
            abort(404)
        saved = connection.execute("SELECT 1 FROM saved_tutors WHERE student_id=? AND teacher_id=?", (g.user["id"], teacher_id)).fetchone()
        if saved:
            connection.execute("DELETE FROM saved_tutors WHERE student_id=? AND teacher_id=?", (g.user["id"], teacher_id))
        else:
            connection.execute("INSERT INTO saved_tutors(student_id,teacher_id,created_at) VALUES(?,?,?)", (g.user["id"], teacher_id, now_iso()))
    if request.accept_mimetypes.best == "application/json":
        return {"saved": not bool(saved)}
    flash("Saved tutors updated.", "success")
    return redirect(request.referrer or url_for("tutor_matches"))


@app.post("/bookings")
@require_role("student")
def create_booking():
    try:
        teacher_id = int(request.form.get("teacher_id", ""))
    except ValueError:
        abort(400)
    kind = request.form.get("session_type", "regular")
    day = request.form.get("session_date", "")
    slot = request.form.get("time_slot", "")
    try:
        session_day = date.fromisoformat(day)
        duration = int(request.form.get("duration_minutes", "60"))
    except ValueError:
        flash("Choose a valid class date and duration.", "error")
        return redirect(url_for("tutor_detail", teacher_id=teacher_id))
    if session_day < date.today() or duration not in {30, 45, 60, 90} or kind not in {"demo", "regular"}:
        flash("Check the class date, type, and duration.", "error")
        return redirect(url_for("tutor_detail", teacher_id=teacher_id))
    tutor = None
    with db() as connection:
        row = connection.execute("SELECT profile_json,verification_status FROM profiles WHERE user_id=?", (teacher_id,)).fetchone()
        if row and row["verification_status"] == "approved":
            tutor = json.loads(row["profile_json"])
        if not tutor:
            abort(404)
        if slot not in tutor.get("availability", []):
            flash("Choose a time the tutor has listed.", "error")
            return redirect(url_for("tutor_detail", teacher_id=teacher_id))
        if kind == "demo" and not tutor.get("offers_demo"):
            flash("This tutor does not currently offer demo classes.", "error")
            return redirect(url_for("tutor_detail", teacher_id=teacher_id))
        if kind == "demo":
            duration = int(tutor.get("demo_minutes", 30))
        amount = float(tutor.get("demo_price", 0)) if kind == "demo" else float(tutor["hourly_rate"]) * duration / 60
        cursor = connection.execute(
            """INSERT INTO bookings(student_id,teacher_id,session_type,session_date,time_slot,duration_minutes,amount,status,payment_status,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (g.user["id"], teacher_id, kind, session_day.isoformat(), slot, duration, amount, "requested", "pending" if amount > 0 else "not_required", now_iso()),
        )
        booking_id = cursor.lastrowid
    notify(teacher_id, "New booking request", f"A student requested a {kind} class on {session_day.isoformat()}.")
    flash("Booking request sent. Continue in Mentora chat.", "success")
    return redirect(url_for("conversation", booking_id=booking_id))


def fetch_booking(booking_id: int) -> dict:
    with db() as connection:
        row = connection.execute(
            """SELECT b.*, su.email AS student_email, tu.email AS teacher_email,
                      sp.profile_json AS student_profile, tp.profile_json AS teacher_profile
               FROM bookings b JOIN users su ON su.id=b.student_id JOIN users tu ON tu.id=b.teacher_id
               LEFT JOIN profiles sp ON sp.user_id=su.id LEFT JOIN profiles tp ON tp.user_id=tu.id
               WHERE b.id=?""", (booking_id,),
        ).fetchone()
    if not row:
        abort(404)
    booking = dict(row)
    booking["student_profile"] = json.loads(booking["student_profile"] or "{}")
    booking["teacher_profile"] = json.loads(booking["teacher_profile"] or "{}")
    booking["teacher_name"] = booking["teacher_profile"].get("display_name", "Tutor")
    booking["student_name"] = booking["student_profile"].get("name") or booking["student_email"]
    return booking


def is_participant(booking: dict) -> bool:
    return g.user["id"] in (booking["student_id"], booking["teacher_id"])


@app.get("/messages/<int:booking_id>")
@require_login
def conversation(booking_id: int):
    booking = fetch_booking(booking_id)
    if not is_participant(booking):
        abort(403)
    with db() as connection:
        messages = connection.execute(
            """SELECT m.*,u.role FROM messages m JOIN users u ON u.id=m.sender_id
               WHERE m.booking_id=? ORDER BY m.created_at""", (booking_id,),
        ).fetchall()
    return render_template("conversation.html", booking=booking, messages=[dict(item) for item in messages])


@app.post("/messages/<int:booking_id>")
@require_login
def send_message(booking_id: int):
    booking = fetch_booking(booking_id)
    if not is_participant(booking):
        abort(403)
    body = request.form.get("body", "").strip()
    if not body or len(body) > 2000:
        flash("Enter a message of 1–2,000 characters.", "error")
        return redirect(url_for("conversation", booking_id=booking_id))
    kind = "text"
    if g.user["role"] == "teacher" and body.lower().startswith(("https://", "http://")):
        kind = "video_link"
    with db() as connection:
        connection.execute("INSERT INTO messages(booking_id,sender_id,body,kind,created_at) VALUES(?,?,?,?,?)", (booking_id, g.user["id"], body, kind, now_iso()))
    recipient = booking["student_id"] if g.user["id"] == booking["teacher_id"] else booking["teacher_id"]
    notify(recipient, "New Mentora message", f"You have a new message about booking #{booking_id}.")
    return redirect(url_for("conversation", booking_id=booking_id))


@app.route("/checkout/<int:booking_id>", methods=["GET", "POST"])
@require_role("student")
def checkout(booking_id: int):
    booking = fetch_booking(booking_id)
    if booking["student_id"] != g.user["id"]:
        abort(403)
    if booking["status"] != "awaiting_payment" or booking["amount"] <= 0:
        flash("This booking is not waiting for payment.", "info")
        return redirect(url_for("conversation", booking_id=booking_id))
    if request.method == "POST":
        with db() as connection:
            connection.execute("UPDATE bookings SET payment_status='demo_paid',status='confirmed' WHERE id=?", (booking_id,))
        notify(booking["teacher_id"], "Booking paid and confirmed", f"Booking #{booking_id} is confirmed in the demo checkout.")
        flash("Demo payment recorded. No real money was charged.", "success")
        return redirect(url_for("dashboard"))
    return render_template("checkout.html", booking=booking)


@app.post("/bookings/<int:booking_id>/action")
@require_login
def booking_action(booking_id: int):
    booking = fetch_booking(booking_id)
    if not is_participant(booking):
        abort(403)
    action = request.form.get("action", "")
    if action == "accept" and g.user["id"] == booking["teacher_id"] and booking["status"] == "requested":
        status = "awaiting_payment" if booking["amount"] > 0 else "confirmed"
        payment_status = "pending" if booking["amount"] > 0 else "not_required"
        with db() as connection:
            connection.execute("UPDATE bookings SET status=?,payment_status=? WHERE id=?", (status, payment_status, booking_id))
        notify(booking["student_id"], "Tutor accepted your request", "Continue to checkout if payment is due.")
        flash("Request accepted.", "success")
    elif action == "decline" and g.user["id"] == booking["teacher_id"] and booking["status"] == "requested":
        with db() as connection:
            connection.execute("UPDATE bookings SET status='declined' WHERE id=?", (booking_id,))
        notify(booking["student_id"], "Booking request declined", "You can browse other tutors and try again.")
        flash("Request declined.", "info")
    elif action == "accept_reschedule" and g.user["id"] == booking["teacher_id"] and booking["status"] == "reschedule_requested":
        next_status = booking.get("reschedule_previous_status") or ("confirmed" if booking["payment_status"] in {"demo_paid", "not_required"} else "awaiting_payment")
        with db() as connection:
            connection.execute(
                """UPDATE bookings SET session_date=requested_date,time_slot=requested_slot,status=?,
                   requested_date=NULL,requested_slot=NULL,reschedule_previous_status=NULL WHERE id=?""",
                (next_status, booking_id),
            )
        notify(booking["student_id"], "Reschedule accepted", "Your class time was updated.")
        flash("New class time accepted.", "success")
    elif action == "decline_reschedule" and g.user["id"] == booking["teacher_id"] and booking["status"] == "reschedule_requested":
        previous = booking.get("reschedule_previous_status") or "confirmed"
        with db() as connection:
            connection.execute("UPDATE bookings SET status=?,requested_date=NULL,requested_slot=NULL,reschedule_previous_status=NULL WHERE id=?", (previous, booking_id))
        notify(booking["student_id"], "Reschedule not accepted", "Your existing booking time remains in place.")
        flash("Reschedule declined; original time remains.", "info")
    elif action == "cancel" and booking["status"] in {"requested", "awaiting_payment", "confirmed", "reschedule_requested"}:
        with db() as connection:
            connection.execute("UPDATE bookings SET status='cancelled' WHERE id=?", (booking_id,))
        other = booking["teacher_id"] if g.user["id"] == booking["student_id"] else booking["student_id"]
        notify(other, "Booking cancelled", f"Booking #{booking_id} was cancelled.")
        flash("Booking cancelled.", "info")
    elif action == "complete" and g.user["id"] == booking["teacher_id"] and booking["status"] == "confirmed":
        with db() as connection:
            connection.execute("UPDATE bookings SET status='completed' WHERE id=?", (booking_id,))
        notify(booking["student_id"], "Class completed", "You can now leave feedback about this tutor match.")
        flash("Class marked complete.", "success")
    else:
        abort(400)
    return redirect(url_for("dashboard"))


@app.post("/bookings/<int:booking_id>/reschedule")
@require_login
def reschedule_booking(booking_id: int):
    booking = fetch_booking(booking_id)
    if not is_participant(booking) or booking["status"] not in {"awaiting_payment", "confirmed"}:
        abort(403)
    requested_date = request.form.get("session_date", "")
    requested_slot = request.form.get("time_slot", "")
    try:
        chosen_date = date.fromisoformat(requested_date)
    except ValueError:
        flash("Choose a valid new date.", "error")
        return redirect(url_for("dashboard"))
    tutor = get_profile(booking["teacher_id"])
    if chosen_date < date.today() or requested_slot not in tutor.get("availability", []):
        flash("Choose a future date and a time listed by the tutor.", "error")
        return redirect(url_for("dashboard"))
    with db() as connection:
        connection.execute(
            """UPDATE bookings SET status='reschedule_requested',requested_date=?,requested_slot=?,
               reschedule_previous_status=? WHERE id=?""",
            (chosen_date.isoformat(), requested_slot, booking["status"], booking_id),
        )
    other = booking["teacher_id"] if g.user["id"] == booking["student_id"] else booking["student_id"]
    notify(other, "Reschedule request", f"A new time was requested for booking #{booking_id}.")
    flash("Reschedule request sent in Mentora.", "success")
    return redirect(url_for("conversation", booking_id=booking_id))


@app.post("/bookings/<int:booking_id>/review")
@require_role("student")
def review_booking(booking_id: int):
    booking = fetch_booking(booking_id)
    if booking["student_id"] != g.user["id"] or booking["status"] != "completed":
        abort(403)
    try:
        rating = int(request.form.get("rating", ""))
    except ValueError:
        abort(400)
    if rating not in range(1, 6):
        abort(400)
    comment = request.form.get("comment", "").strip()[:1000]
    with db() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO reviews(booking_id,student_id,teacher_id,rating,comment,created_at) VALUES(?,?,?,?,?,?)",
            (booking_id, g.user["id"], booking["teacher_id"], rating, comment, now_iso()),
        )

    notify(booking["teacher_id"], "New student review", "A student left feedback after a class.")
    flash("Thanks for sharing feedback.", "success")
    return redirect(url_for("dashboard"))


@app.post("/tutors/<int:teacher_id>/report")
@require_login
def report_tutor(teacher_id: int):
    if teacher_id == g.user["id"]:
        abort(400)
    reason = request.form.get("reason", "").strip()[:500]
    if not reason:
        flash("Add a short reason for the report.", "error")
        return redirect(request.referrer or url_for("dashboard"))
    with db() as connection:
        connection.execute("INSERT INTO reports(reporter_id,teacher_id,reason,created_at) VALUES(?,?,?,?)", (g.user["id"], teacher_id, reason, now_iso()))
    flash("Report sent to the Mentora review queue.", "success")
    return redirect(request.referrer or url_for("dashboard"))


@app.post("/tutors/<int:teacher_id>/block")
@require_login
def block_tutor(teacher_id: int):
    if teacher_id == g.user["id"]:
        abort(400)
    with db() as connection:
        connection.execute("INSERT OR IGNORE INTO blocks(blocker_id,blocked_id,created_at) VALUES(?,?,?)", (g.user["id"], teacher_id, now_iso()))
    flash("This tutor will be hidden from your recommendations.", "info")
    return redirect(url_for("dashboard"))


@app.get("/admin")
@require_role("admin")
def admin_dashboard():
    with db() as connection:
        pending = connection.execute(
            """SELECT u.id,u.email,p.profile_json FROM users u JOIN profiles p ON p.user_id=u.id
               WHERE u.role='teacher' AND p.verification_status='pending' ORDER BY p.updated_at"""
        ).fetchall()
        reports = connection.execute("SELECT * FROM reports WHERE status='open' ORDER BY created_at").fetchall()
    return render_template(
        "admin.html",
        pending=[{"id": row["id"], "email": row["email"], "profile": json.loads(row["profile_json"])} for row in pending],
        reports=[dict(row) for row in reports],
    )


@app.post("/admin/tutors/<int:teacher_id>/verify")
@require_role("admin")
def verify_tutor(teacher_id: int):
    decision = request.form.get("decision")
    if decision not in {"approved", "rejected"}:
        abort(400)
    with db() as connection:
        connection.execute("UPDATE profiles SET verification_status=?,updated_at=? WHERE user_id=?", (decision, now_iso(), teacher_id))
    notify(teacher_id, "Tutor profile review complete", f"Your profile was {decision} by the Mentora review team.")
    flash(f"Tutor profile {decision}.", "success")
    return redirect(url_for("admin_dashboard"))


@app.post("/admin/reports/<int:report_id>/close")
@require_role("admin")
def close_report(report_id: int):
    with db() as connection:
        connection.execute("UPDATE reports SET status='reviewed' WHERE id=?", (report_id,))
    flash("Report marked reviewed.", "success")
    return redirect(url_for("admin_dashboard"))


@app.get("/matching-method")
def matching_method():
    bundle = load_trained_model()
    return render_template("model.html", report=bundle.get("evaluation") if bundle else None)


@app.get("/about")
def about():
    return render_template("about.html")


@app.get("/health")
def health():
    return {"status": "ok", "app": "Mentora"}


init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5001")), debug=os.environ.get("FLASK_DEBUG") == "1")
