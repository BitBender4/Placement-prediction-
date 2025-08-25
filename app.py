import os
import re
import json
import datetime
import traceback
import matplotlib.pyplot as plt
# from datetime import datetime
import io
from flask import Flask, render_template, request, send_file, url_for
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import numpy as np
import joblib
import tensorflow as tf
import requests

# ---- Extra libs for resume parsing ----
from pdfminer.high_level import extract_text as pdf_extract_text
import docx
from fuzzywuzzy import fuzz

from main import OPENAI_API_KEY

# =========================
# Paths & App Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXT = {"pdf", "docx", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
CORS(app)

OPENROUTER_API_KEY = "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"

# For dashboard persistence
# DATA_FILE = os.path.join(BASE_DIR, "predictions.json")

# =========================
# Model Loading (single, safe)
# =========================
try:
    placement_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "placement_model_tf.h5"))
    salary_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "salary_model_tf.h5"), compile=False)
    placement_scaler = joblib.load(os.path.join(MODEL_DIR, "scalar.pkl"))
    salary_scaler = joblib.load(os.path.join(MODEL_DIR, "salary_scaler.pkl"))
    print("✅ Models and scalers loaded successfully")
except Exception as e:
    print(f"❌ Error loading models/scalers: {e}")
    placement_model = salary_model = placement_scaler = salary_scaler = None

# =========================
# Helpers: dashboard data
# =========================
DATA_FILE = "predictions.json"


def save_prediction(data):
    """Append new prediction to JSON file"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(data)

    with open(DATA_FILE, "w") as f:
        json.dump(history, f, indent=4)


def load_predictions():
    """Load all predictions from file"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

# =========================
# Placement/Salary Prediction
# =========================
def run_predictions(form_data: dict):
    placement_prediction = salary_prediction = None
    skill_suggestions = []

    if not placement_model or not salary_model or not placement_scaler or not salary_scaler:
        return None, None, ["Models not loaded"]

    try:
        # Placement features
        placement_keys = [
            'College_ID', 'IQ', 'Prev_Sem_Result', 'CGPA',
            'Academic_Performance', 'Internship_Experience',
            'Extra_Curricular_Score', 'Communication_Skills',
            'Projects_Completed'
        ]
        placement_features = np.array([_safe_float(form_data.get(k, 0)) for k in placement_keys]).reshape(1, -1)
        pf_scaled = placement_scaler.transform(placement_features)
        placed_prob = float(placement_model.predict(pf_scaled)[0][0])
        placed = placed_prob >= 0.3
        placement_prediction = "Placed" if placed else "Not Placed"

        # Salary features
        salary_keys = [
            'Name of Student','Roll No.','No_of_DSA_questions', 'CGPA', 'Knows_ML',
            'Knows_DSA', 'Knows_Python', 'Knows_JavaScript',
            'Knows_HTML', 'Knows_CSS', 'Knows_Cricket',
            'Knows_Dance', 'Participated_in_College_Fest',
            'Was_in_Coding_Club', 'No_of_backlogs',
            'Interview_Room_Temperature', 'Age_of_Candidate',
            'Branch_of_Engineering'
        ]
        salary_features = np.array([_safe_float(form_data.get(k, 0)) for k in salary_keys]).reshape(1, -1)
        sf_scaled = salary_scaler.transform(salary_features)

        if placed:
            salary = float(salary_model.predict(sf_scaled)[0][0])
            salary_prediction = f"{round(salary, 2)} LPA"

            # Skill suggestions
            skills = {
                "Machine Learning": _safe_int(form_data.get('Knows_ML', 0)),
                "DSA": _safe_int(form_data.get('Knows_DSA', 0)),
                "Python": _safe_int(form_data.get('Knows_Python', 0)),
                "JavaScript": _safe_int(form_data.get('Knows_JavaScript', 0)),
                "HTML": _safe_int(form_data.get('Knows_HTML', 0)),
                "CSS": _safe_int(form_data.get('Knows_CSS', 0)),
                "Coding Club": _safe_int(form_data.get('Was_in_Coding_Club', 0)),
                "Fest Participation": _safe_int(form_data.get('Participated_in_College_Fest', 0)),
                "Communication": _safe_float(form_data.get('Communication_Skills', 0.0)),
            }
            for skill, val in skills.items():
                if skill == "Communication":
                    if val < 6:
                        skill_suggestions.append("Improve communication skills")
                else:
                    if val == 0:
                        skill_suggestions.append(f"Consider learning {skill}")
        else:
            salary_prediction = "Not Applicable (Not Placed)"
            if not skill_suggestions:
                skill_suggestions = ["Focus on improving placement chances first."]

        return placement_prediction, salary_prediction, skill_suggestions
    except Exception:
        traceback.print_exc()
        return None, None, ["Prediction error"]

# =========================
# Resume Analyzer (merged from resume.py)
# =========================
SKILLS = [
    "python","java","c++","c","javascript","html","css","react","node.js","django","flask",
    "tensorflow","keras","pytorch","scikit-learn","pandas","numpy","sql","mongodb","git",
    "docker","kubernetes","aws","azure","gcp","data structures","algorithms","machine learning"
]
REQUIRED_SKILLS_FOR_PLACEMENT = ["data structures", "algorithms", "python", "git"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

def extract_text(path):
    ext = path.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        try:
            return pdf_extract_text(path)
        except Exception:
            return ""
    elif ext == "docx":
        return extract_text_from_docx(path)
    elif ext == "txt":
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    return ""

def normalize_text(text):
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.lower()

def find_skills(text, skills_list, threshold=85):
    found = {}
    for s in skills_list:
        if re.search(r"\b" + re.escape(s) + r"\b", text):
            found[s] = {"match_type": "exact", "score": 100}
        else:
            r = fuzz.partial_ratio(s, text)
            if r >= threshold:
                found[s] = {"match_type": "fuzzy", "score": int(r)}
    return found

def extract_contact_indicators(text):
    has_email = bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text))
    digits = re.sub(r"[^0-9]", "", text)
    has_phone = len(digits) >= 10
    has_github = "github.com" in text
    return {"has_email": has_email, "has_phone": has_phone, "has_github": has_github}

def score_resume(found_skills, required_skills, contact_flags, raw_text):
    req = set([r.lower() for r in required_skills])
    found = set([k.lower() for k in found_skills.keys()])
    covered = req.intersection(found)
    skill_score = int((len(covered) / max(1, len(req))) * 60)
    contact_score = 0
    contact_score += 10 if contact_flags.get("has_email") else 0
    contact_score += 5 if contact_flags.get("has_phone") else 0
    contact_score += 5 if contact_flags.get("has_github") else 0
    exp_points = 0
    if re.search(r"\bintern(ship)?s?\b", raw_text):
        exp_points += 10
    if re.search(r"\bproject(s)?\b", raw_text):
        exp_points += 5
    total = skill_score + contact_score + min(exp_points, 20)
    return max(0, min(100, total))

def generate_suggestions(found_skills, required_skills):
    found = set([k.lower() for k in found_skills.keys()])
    missing = [r for r in required_skills if r.lower() not in found]
    suggestions = []
    for m in missing:
        suggestions.append({
            "skill": m,
            "why": "Needed for common placement interviews and coding rounds.",
            "how_to_learn": [
                f"Practice {m} problems on platforms like LeetCode/HackerRank.",
                f"Add a small project that demonstrates {m} to your resume."
            ]
        })
    suggestions.append({"format": "Add GitHub link and 2–3 project bullets with measurable impact."})
    suggestions.append({"format": "Mention internships and role/responsibilities under Experience clearly."})
    return suggestions


#===============================
#Github analyzer
#=================================

GITHUB_API = "https://api.github.com/users/"

def fetch_user_and_repos(username):
    user_resp = requests.get(GITHUB_API + username)
    repos_resp = requests.get(GITHUB_API + username + "/repos?per_page=200")

    if user_resp.status_code != 200:
        return None, None, f"User '{username}' not found (status {user_resp.status_code})."

    user = user_resp.json()
    repos = []
    if repos_resp.status_code == 200:
        repos = repos_resp.json()
    else:
        return user, [], f"Could not fetch repos (status {repos_resp.status_code})."

    return user, repos, None

def aggregate_repo_stats(repos):
    total_stars = 0
    total_forks = 0
    total_watchers = 0
    language_count = {}
    most_popular = None

    for repo in repos:
        stars = repo.get("stargazers_count", 0) or 0
        forks = repo.get("forks_count", 0) or 0
        watchers = repo.get("watchers_count", 0) or 0
        total_stars += stars
        total_forks += forks
        total_watchers += watchers

        lang = repo.get("language")
        if lang:
            language_count[lang] = language_count.get(lang, 0) + 1

        if (most_popular is None) or (stars > most_popular.get("stargazers_count", 0)):
            most_popular = repo

    return {
        "total_stars": total_stars,
        "total_forks": total_forks,
        "total_watchers": total_watchers,
        "languages": language_count,
        "most_popular": most_popular
    }

def get_recent_repos(repos, n=5):
    repos_with_date = [r for r in repos if r.get("pushed_at")]
    repos_with_date.sort(key=lambda r: r["pushed_at"], reverse=True)
    recent = repos_with_date[:n]
    for r in recent:
        pushed = r.get("pushed_at")
        try:
            r["pushed_at_fmt"] = datetime.strptime(pushed, "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d, %Y")
        except:
            r["pushed_at_fmt"] = pushed
    return recent

# =========================
# Routes
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():
    history = load_predictions() or []  # ensure history is at least an empty list
    total = len(history)

    # Safely count placed / not placed
    placed = sum(1 for h in history if h.get('status') == "Placed")
    not_placed = total - placed

    # Safely calculate average salary
    salaries = [h.get('expected_salary', 0) for h in history if h.get('expected_salary') is not None]
    avg_salary = round(sum(salaries) / len(salaries), 2) if salaries else 0

    stats = {
        "total_students": total,
        "placement_rate": round((placed / total) * 100, 2) if total > 0 else 0,
        "average_salary": avg_salary,
        "accuracy": 85
    }

    charts = {
        "placement_distribution": {
            "labels": ["Placed", "Not Placed"],
            "values": [placed, not_placed]
        },
        "salary_distribution": {
            "labels": [h.get("cgpa", 0) for h in history if h.get("cgpa") is not None],
            "values": [h.get("expected_salary", 0) for h in history if h.get("expected_salary") is not None]
        }
    }

    return render_template("index1.html", stats=stats, charts=charts)

# ----- API Prediction (AJAX) -----
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True) or {}
    placement_prediction, salary_prediction, skill_suggestions = run_predictions(data)

    if placement_prediction is None:
        placement_prediction = "Prediction Failed"
    if salary_prediction is None:
        salary_prediction = "N/A"

    is_placed = (str(placement_prediction).lower() == "placed")
    placement_probability = 1.0 if is_placed else 0.0

    predicted_salary_value = 0.0
    if isinstance(salary_prediction, str) and "lpa" in salary_prediction.lower():
        try:
            predicted_salary_value = float(salary_prediction.lower().replace("lpa", "").strip())
        except ValueError:
            predicted_salary_value = 0.0

    record = {
        "timestamp": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "cgpa": data.get("CGPA", "N/A"),
        "technical_skills": data.get("Knows_Python", "N/A"),
        "projects": data.get("Projects_Completed", "N/A"),
        "placement_probability": round(placement_probability * 100, 2),
        "expected_salary": predicted_salary_value,
        "status": placement_prediction,
        "skill_suggestions": skill_suggestions
    }
    save_prediction(record)

    return jsonify({
        "success": True,
        "placement_prediction": placement_prediction,
        "salary_prediction": salary_prediction,
        "skill_suggestions": skill_suggestions,
        "prediction": record
    })

# ----- Dashboard -----
@app.route('/dashboard')
def dashboard():
    history = load_predictions() or []
    total = len(history)

    # Use .get() to avoid KeyError
    placed = sum(1 for h in history if h.get('status') == "Placed")
    not_placed = total - placed

    # Safely calculate average salary
    salaries = [h.get('expected_salary', 0) for h in history if h.get('expected_salary') is not None]
    avg_salary = round(sum(salaries) / len(salaries), 2) if salaries else 0

    stats = {
        "total_students": total,
        "placement_rate": round((placed / total) * 100, 2) if total > 0 else 0,
        "average_salary": avg_salary,
        "accuracy": 85
    }

    charts = {
        "placement_distribution": {
            "labels": ["Placed", "Not Placed"],
            "values": [placed, not_placed]
        },
        "salary_distribution": {
            "labels": [h.get("cgpa", 0) for h in history if h.get("cgpa") is not None],
            "values": [h.get("expected_salary", 0) for h in history if h.get("expected_salary") is not None]
        }
    }

    return render_template("dashboard.html", stats=stats, charts=charts)

@app.route('/api/dashboard-data')
def api_dashboard_data():
    predictions = load_predictions()
    total_predictions = len(predictions)
    placed_count = sum(1 for p in predictions if p['status'] == 'Placed')
    placement_rate = round((placed_count / total_predictions) * 100, 2) if total_predictions > 0 else 0
    avg_salary = round(sum(p['expected_salary'] for p in predictions) / total_predictions, 2) if total_predictions > 0 else 0
    accuracy = 85

    skill_suggestions = []
    for p in predictions[-1:]:
        skill_suggestions.extend(p.get("skill_suggestions", []))

    recent_predictions = predictions[-5:]
    salary_bins = [p['expected_salary'] for p in predictions if p['expected_salary'] > 0]
    placement_distribution = {'Placed': placed_count, 'Not Placed': total_predictions - placed_count}

    skills_success = {}
    for p in predictions:
        if "skills" in p:
            for skill, value in p['skills'].items():
                if skill not in skills_success:
                    skills_success[skill] = {'total': 0, 'placed': 0}
                skills_success[skill]['total'] += 1
                if p['status'] == 'Placed':
                    skills_success[skill]['placed'] += 1
    skills_analysis = {k: round(v['placed'] / v['total'] * 100, 2) for k, v in skills_success.items()} if skills_success else {}

    activity = {}
    for p in predictions:
        date = p['timestamp'].split(' ')[0]
        activity[date] = activity.get(date, 0) + 1
    activity_over_time = [{'timestamp': k, 'count': v} for k, v in sorted(activity.items())]

    data = {
        "total_predictions": total_predictions,
        "placement_rate": placement_rate,
        "avg_salary": avg_salary,
        "accuracy": accuracy,
        "recent_predictions": recent_predictions,
        "salary_distribution": salary_bins,
        "placement_distribution": placement_distribution,
        "skills_analysis": skills_analysis,
        "activity_over_time": activity_over_time,
        "feature_importance": {"CGPA": 0.35, "Skills": 0.25, "Projects": 0.20, "Backlogs": 0.10, "Internships": 0.10},
        "skill_suggestions": skill_suggestions
    }
    return jsonify(data)

# ----- Chatbot -----
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route("/get", methods=["POST"])
def get_bot_response():
    try:
        data = request.get_json()
        user_input = (data or {}).get("msg", "").strip()

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",  # ✅ OpenAI model name
            "messages": [{"role": "user", "content": user_input}]
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if resp.status_code == 401:
            return jsonify({"response": "❌ Unauthorized: Invalid or expired OpenAI API key."})

        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})


# ----- Resume pages -----
@app.route('/resume_index')
def resume_index():
    return render_template('resume_index.html')

@app.route('/github_index')
def github_index():
    return render_template('github_index.html')

# *********** MERGED RESUME ANALYZE ROUTE (fixes contact undefined) ***********
@app.route('/analyze', methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return redirect(url_for("resume_index"))

    f = request.files["resume"]
    if f.filename.strip() == "":
        return redirect(url_for("resume_index"))
    if not allowed_file(f.filename):
        return render_template("resume_index.html", error="File type not allowed. Use PDF/DOCX/TXT.")

    filename = secure_filename(f.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)

    raw = extract_text(path)
    text = normalize_text(raw)

    found_skills = find_skills(text, SKILLS, threshold=85)
    contact = extract_contact_indicators(text) or {}
    score = score_resume(found_skills, REQUIRED_SKILLS_FOR_PLACEMENT, contact, text)
    suggestions = generate_suggestions(found_skills, REQUIRED_SKILLS_FOR_PLACEMENT)

    # IMPORTANT: pass ALL variables your template expects -> no 'contact undefined'
    return render_template(
        "resume_results.html",
        score=score,
        found_skills=found_skills,
        contact=contact,
        suggestions=suggestions,
        filename=filename
    )

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


#=================
#github routes
#==================
@app.route("/github", methods=["GET", "POST"])
def github():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        if not username:
            return render_template("github_index.html", error="Please enter a GitHub username.")

        user, repos, err = fetch_user_and_repos(username)
        if err:
            return render_template("github_index.html", error=err)

        agg = aggregate_repo_stats(repos)
        recent = get_recent_repos(repos, n=5)

        created_at = user.get("created_at")
        try:
            created_fmt = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").strftime("%b %d, %Y")
        except:
            created_fmt = created_at

        data = {
            "username": username,
            "name": user.get("name") or username,
            "bio": user.get("bio"),
            "location": user.get("location"),
            "created_at": created_fmt,
            "avatar_url": user.get("avatar_url"),
            "profile_url": user.get("html_url"),
            "followers": user.get("followers", 0),
            "following": user.get("following", 0),
            "public_repos": user.get("public_repos", 0),
            "total_stars": agg["total_stars"],
            "total_forks": agg["total_forks"],
            "total_watchers": agg["total_watchers"],
            "languages": agg["languages"],
            "most_popular": agg["most_popular"],
            "recent_repos": recent
        }

        return render_template("github_result.html", data=data)

    return render_template("github_index.html")

@app.route("/lang-chart.png")
def lang_chart():
    username = request.args.get("username", "")
    if not username:
        fig = plt.figure(figsize=(4, 4))
        plt.text(0.5, 0.5, 'No user', ha='center')
    else:
        user, repos, err = fetch_user_and_repos(username)
        if err:
            fig = plt.figure(figsize=(4, 4))
            plt.text(0.5, 0.5, 'Error', ha='center')
        else:
            agg = aggregate_repo_stats(repos)
            languages = agg["languages"]
            if not languages:
                fig = plt.figure(figsize=(4, 4))
                plt.text(0.5, 0.5, 'No languages', ha='center')
            else:
                labels = list(languages.keys())
                sizes = list(languages.values())
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
                ax.axis('equal')
                plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close('all')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route("/about")
def about():
    return render_template("about.html")



# =========================
# Main
# =========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)