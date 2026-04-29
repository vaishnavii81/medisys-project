
from flask import (Flask, request, session, redirect,
                   render_template_string, jsonify, make_response)
import pandas as pd
import sqlite3, csv, io, hashlib
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

app = Flask(__name__)
# app.secret_key = ""

def get_db():
    conn = sqlite3.connect("medisys.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db(); c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT UNIQUE, password TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS patients(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL, age INTEGER, gender TEXT,
        contact TEXT, symptoms TEXT, diagnosis TEXT,
        risk_level TEXT, date_of_visit TEXT, notes TEXT, created_by TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS appointments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT, doctor TEXT, date TEXT, time TEXT,
        reason TEXT, status TEXT DEFAULT 'Scheduled', created_by TEXT)""")
    conn.commit(); conn.close()

init_db()

data = pd.read_csv("heart_disease.csv")
data.columns = data.columns.str.strip()
X = data.drop(["target_binary","num"], axis=1)
y = data["target_binary"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

questions = [
    ("age","Age","e.g. 55"),
    ("sex","Sex (1=Male, 0=Female)","1 or 0"),
    ("cp","Chest Pain Type (0-3)","0-3"),
    ("trestbps","Resting Blood Pressure","e.g. 130"),
    ("chol","Cholesterol (mg/dl)","e.g. 240"),
    ("fbs","Fasting Blood Sugar >120 (1/0)","1 or 0"),
    ("restecg","Resting ECG (0-2)","0-2"),
    ("thalach","Max Heart Rate","e.g. 150"),
    ("exang","Exercise Induced Angina (1/0)","1 or 0"),
    ("oldpeak","ST Depression","e.g. 1.5"),
    ("slope","Slope of ST (0-2)","0-2"),
    ("ca","Major Vessels (0-3)","0-3"),
    ("thal","Thalassemia (0-3)","0-3"),
]

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

CHAT_RULES = [
    (["hello","hi","hey"],"Hi! I am MediBot. Ask me about symptoms, heart health, or how to use MediSys."),
    (["symptom","symptoms"],"Common heart symptoms include chest pain, shortness of breath, palpitations, and fatigue. Use the Risk Prediction tool for assessment."),
    (["precaution","prevent"],"Prevention tips: regular exercise, low-sodium diet, avoid smoking, manage stress, annual check-ups."),
    (["diet","food","eat"],"Heart-healthy foods: oats, salmon, nuts, berries, leafy greens, olive oil. Limit red meat and processed food."),
    (["exercise","workout"],"Aim for 150 min of moderate aerobic activity per week. Walking, cycling, and swimming are great."),
    (["appointment","book","schedule"],"Go to the Appointments section in the sidebar to schedule a visit."),
    (["patient","record","add"],"Add patient records from the Patients section with name, age, symptoms, diagnosis and more."),
    (["predict","prediction","risk"],"Head to Risk Prediction to enter clinical values and get an AI-based heart disease assessment."),
    (["export","download","csv"],"Export all patient records as CSV from the Patients section using the Export button."),
    (["help","guide","how"],"MediSys features: Dashboard, Patient Records, Risk Prediction, Appointments, MediBot, and Reports."),
    (["bp","blood pressure"],"Normal BP is below 120/80 mmHg. High BP is a major risk factor for heart disease."),
    (["cholesterol","chol"],"Normal total cholesterol is below 200 mg/dL. Above 240 significantly raises heart disease risk."),
]

def chatbot_reply(msg):
    ml = msg.lower()
    for kws, reply in CHAT_RULES:
        if any(k in ml for k in kws):
            return reply
    return "Try asking about symptoms, prevention, diet, appointments, or how to use MediSys."

CSS = """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
:root{--bg:#f0f4f8;--sidebar:#0f2a4a;--sidebar-hover:#1a3d6b;--accent:#2563eb;--accent2:#10b981;
  --danger:#ef4444;--warn:#f59e0b;--card:#fff;--text:#1e293b;--muted:#64748b;
  --border:#e2e8f0;--radius:14px;--shadow:0 2px 16px rgba(15,42,74,.09);}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);display:flex;min-height:100vh;}
.sidebar{width:250px;min-height:100vh;background:var(--sidebar);display:flex;flex-direction:column;
  position:fixed;top:0;left:0;z-index:100;box-shadow:4px 0 20px rgba(0,0,0,.15);}
.sidebar-logo{padding:24px 20px 20px;border-bottom:1px solid rgba(255,255,255,.08);
  display:flex;align-items:center;gap:12px;}
.logo-icon{width:40px;height:40px;background:var(--accent);border-radius:10px;
  display:flex;align-items:center;justify-content:center;font-size:20px;}
.sidebar-logo span{color:#fff;font-weight:700;font-size:18px;}
.sidebar-logo small{color:rgba(255,255,255,.4);font-size:11px;display:block;}
.sidebar-nav{flex:1;padding:16px 12px;}
.nav-section{font-size:10px;font-weight:700;color:rgba(255,255,255,.3);letter-spacing:1.5px;
  text-transform:uppercase;padding:8px 8px 4px;margin-top:8px;}
.nav-item{display:flex;align-items:center;gap:12px;padding:11px 12px;border-radius:10px;
  color:rgba(255,255,255,.65);text-decoration:none;font-size:14px;font-weight:500;
  margin-bottom:2px;transition:all .2s;}
.nav-item:hover,.nav-item.active{background:var(--sidebar-hover);color:#fff;}
.nav-item.active{background:var(--accent);}
.nav-item i{width:18px;text-align:center;font-size:14px;}
.sidebar-user{padding:16px;border-top:1px solid rgba(255,255,255,.08);
  display:flex;align-items:center;gap:10px;}
.user-avatar{width:36px;height:36px;border-radius:50%;background:var(--accent2);
  display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;
  font-size:14px;flex-shrink:0;}
.user-info span{color:#fff;font-size:13px;font-weight:600;display:block;}
.user-info small{color:rgba(255,255,255,.4);font-size:11px;}
.logout-btn{margin-left:auto;color:rgba(255,255,255,.4);font-size:16px;text-decoration:none;}
.logout-btn:hover{color:var(--danger);}
.main{margin-left:250px;flex:1;padding:28px 32px;min-height:100vh;}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;}
.topbar h1{font-family:'DM Serif Display';font-size:26px;}
.topbar p{color:var(--muted);font-size:14px;margin-top:2px;}
.card{background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);
  padding:24px;border:1px solid var(--border);}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px;}
.stat-card{background:var(--card);border-radius:var(--radius);padding:20px;
  border:1px solid var(--border);box-shadow:var(--shadow);}
.stat-icon{width:48px;height:48px;border-radius:12px;display:flex;align-items:center;
  justify-content:center;font-size:20px;margin-bottom:12px;}
.stat-value{font-size:28px;font-weight:700;}
.stat-label{font-size:13px;color:var(--muted);margin-top:2px;}
table{width:100%;border-collapse:collapse;font-size:14px;}
th{background:#f8fafc;padding:12px 16px;text-align:left;font-weight:600;font-size:12px;
  color:var(--muted);text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid var(--border);}
td{padding:13px 16px;border-bottom:1px solid #f1f5f9;vertical-align:middle;}
tr:hover td{background:#f8fafc;}
.badge{display:inline-flex;align-items:center;padding:4px 10px;border-radius:20px;font-size:12px;font-weight:600;}
.badge-red{background:#fee2e2;color:#dc2626;}
.badge-green{background:#d1fae5;color:#059669;}
.badge-blue{background:#dbeafe;color:#2563eb;}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
.form-group{display:flex;flex-direction:column;gap:6px;}
.form-group.full{grid-column:1/-1;}
label{font-size:13px;font-weight:600;color:var(--text);}
input,select,textarea{padding:10px 14px;border:1.5px solid var(--border);border-radius:9px;
  font-family:inherit;font-size:14px;background:#fafafa;transition:.2s;outline:none;}
input:focus,select:focus,textarea:focus{border-color:var(--accent);
  box-shadow:0 0 0 3px rgba(37,99,235,.1);background:#fff;}
textarea{resize:vertical;min-height:80px;}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 20px;border-radius:9px;
  font-family:inherit;font-size:14px;font-weight:600;cursor:pointer;text-decoration:none;
  border:none;transition:all .2s;}
.btn-primary{background:var(--accent);color:#fff;}
.btn-primary:hover{background:#1d4ed8;transform:translateY(-1px);}
.btn-success{background:var(--accent2);color:#fff;}
.btn-danger{background:var(--danger);color:#fff;}
.btn-outline{background:transparent;border:1.5px solid var(--border);color:var(--text);}
.btn-outline:hover{border-color:var(--accent);color:var(--accent);}
.btn-sm{padding:6px 12px;font-size:12px;border-radius:7px;}
.search-bar{display:flex;gap:10px;margin-bottom:18px;align-items:center;}
.search-bar input{flex:1;max-width:300px;}
.search-bar select{max-width:160px;}
.chat-window{height:360px;overflow-y:auto;border:1.5px solid var(--border);
  border-radius:12px;padding:16px;background:#f8fafc;display:flex;flex-direction:column;gap:10px;}
.chat-msg{max-width:80%;padding:10px 14px;border-radius:16px;font-size:14px;line-height:1.5;}
.chat-msg.bot{background:#fff;border:1px solid var(--border);
  border-radius:4px 16px 16px 16px;align-self:flex-start;}
.chat-msg.user{background:var(--accent);color:#fff;
  border-radius:16px 4px 16px 16px;align-self:flex-end;}
.chat-input-row{display:flex;gap:8px;margin-top:12px;}
.chat-input-row input{flex:1;}
.alert{padding:12px 16px;border-radius:9px;font-size:14px;margin-bottom:16px;}
.alert-danger{background:#fee2e2;color:#991b1b;border:1px solid #fecaca;}
.alert-success{background:#d1fae5;color:#065f46;border:1px solid #a7f3d0;}
.flex{display:flex;}.gap-2{gap:8px;}.gap-3{gap:12px;}
.justify-between{justify-content:space-between;}.items-center{align-items:center;}
.mb-4{margin-bottom:16px;}.mb-6{margin-bottom:24px;}.mt-4{margin-top:16px;}
.text-muted{color:var(--muted);font-size:13px;}
.section-title{font-size:17px;font-weight:700;margin-bottom:16px;
  display:flex;align-items:center;gap:8px;}
.divider{border:none;border-top:1px solid var(--border);margin:20px 0;}
</style>
"""

def base(content, active="dashboard"):
    user = session.get("user","")
    uname = session.get("uname", user.split("@")[0].title())
    nav = [
        ("dashboard","fa-gauge","Dashboard"),
        ("patients","fa-users","Patients"),
        ("predict","fa-heart-pulse","Risk Prediction"),
    ]
    nav2 = [
        ("appointments","fa-calendar","Appointments"),
        ("chatbot","fa-robot","MediBot"),
        ("reports","fa-file-medical","Reports"),
    ]
    def ni(key,icon,label):
        cls = "active" if key==active else ""
        return f'<a href="/{key}" class="nav-item {cls}"><i class="fa {icon}"></i><span>{label}</span></a>'
    nav_html = "".join(ni(*x) for x in nav)
    nav2_html = "".join(ni(*x) for x in nav2)
    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MediSys</title>{CSS}</head><body>
<aside class="sidebar">
  <div class="sidebar-logo">
    <div class="logo-icon">&#x1F3E5;</div>
    <div><span>MediSys</span><small>Healthcare Management</small></div>
  </div>
  <nav class="sidebar-nav">
    <div class="nav-section">Main</div>{nav_html}
    <div class="nav-section">Tools</div>{nav2_html}
  </nav>
  <div class="sidebar-user">
    <div class="user-avatar">{uname[0].upper()}</div>
    <div class="user-info"><span>{uname}</span><small>{user}</small></div>
    <a href="/logout" class="logout-btn" title="Logout"><i class="fa fa-right-from-bracket"></i></a>
  </div>
</aside>
<main class="main">{content}</main>
</body></html>"""

LOGIN_HTML = """<!DOCTYPE html><html><head>
<meta charset="UTF-8"><title>MediSys Login</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Serif+Display&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;min-height:100vh;display:flex;
  background:linear-gradient(135deg,#0f2a4a 0%,#1a3d6b 50%,#0f4c75 100%);}
.left{flex:1;display:flex;flex-direction:column;justify-content:center;padding:60px;color:#fff;}
.left h1{font-family:'DM Serif Display';font-size:44px;line-height:1.2;margin-bottom:16px;}
.left p{opacity:.7;font-size:16px;max-width:380px;line-height:1.7;}
.feats{margin-top:36px;display:flex;flex-direction:column;gap:14px;}
.feat{display:flex;align-items:center;gap:12px;font-size:14px;opacity:.85;}
.fi{width:32px;height:32px;border-radius:8px;background:rgba(255,255,255,.12);
  display:flex;align-items:center;justify-content:center;}
.right{width:440px;background:#fff;display:flex;align-items:center;justify-content:center;padding:48px;}
.box{width:100%;}
.logo{display:flex;align-items:center;gap:12px;margin-bottom:32px;}
.li{width:44px;height:44px;background:#2563eb;border-radius:12px;
  display:flex;align-items:center;justify-content:center;font-size:22px;}
h2{font-size:24px;font-weight:700;color:#0f2a4a;margin-bottom:6px;}
.sub{color:#64748b;font-size:14px;margin-bottom:28px;}
.fg{margin-bottom:16px;}
label{display:block;font-size:13px;font-weight:600;color:#334155;margin-bottom:6px;}
input{width:100%;padding:11px 14px;border:1.5px solid #e2e8f0;border-radius:9px;
  font-family:inherit;font-size:14px;outline:none;transition:.2s;}
input:focus{border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.1);}
.btn{width:100%;padding:12px;background:#2563eb;color:#fff;border:none;border-radius:9px;
  font-size:15px;font-weight:600;cursor:pointer;font-family:inherit;margin-top:8px;}
.btn:hover{background:#1d4ed8;}
.lnk{text-align:center;margin-top:20px;font-size:14px;color:#64748b;}
.lnk a{color:#2563eb;text-decoration:none;font-weight:600;}
.err{background:#fee2e2;color:#991b1b;border:1px solid #fecaca;
  border-radius:8px;padding:10px 14px;font-size:13px;margin-bottom:16px;}
</style></head><body>
<div class="left">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:40px;">
    <div class="fi" style="background:rgba(255,255,255,.15);font-size:22px;">&#x1F3E5;</div>
    <span style="font-size:24px;font-weight:700;">MediSys</span>
  </div>
  <h1>Advanced Healthcare Management</h1>
  <p>A smart platform for patient data, appointments, and AI-powered heart disease risk prediction.</p>
  <div class="feats">
    <div class="feat"><div class="fi">&#x1F4CB;</div>Patient Records &amp; History</div>
    <div class="feat"><div class="fi">&#x1F916;</div>AI Risk Prediction Engine</div>
    <div class="feat"><div class="fi">&#x1F4C5;</div>Appointment Scheduling</div>
    <div class="feat"><div class="fi">&#x1F4CA;</div>Analytics Dashboard</div>
  </div>
</div>
<div class="right"><div class="box">
  <div class="logo"><div class="li">&#x1F3E5;</div><span style="font-size:22px;font-weight:700;color:#0f2a4a;">MediSys</span></div>
  <h2>Welcome back</h2><p class="sub">Sign in to your account</p>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
  <form method="post">
    <div class="fg"><label>Email</label><input name="email" type="email" placeholder="you@example.com" required></div>
    <div class="fg"><label>Password</label><input name="password" type="password" placeholder="&#x2022;&#x2022;&#x2022;&#x2022;&#x2022;&#x2022;&#x2022;&#x2022;" required></div>
    <button class="btn">Sign In</button>
  </form>
  <div class="lnk">No account? <a href="/signup">Create one</a></div>
</div></div></body></html>"""

SIGNUP_HTML = """<!DOCTYPE html><html><head>
<meta charset="UTF-8"><title>MediSys Signup</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'DM Sans',sans-serif;min-height:100vh;display:flex;align-items:center;
  justify-content:center;background:linear-gradient(135deg,#0f2a4a,#1a3d6b);}
.box{background:#fff;border-radius:18px;padding:44px;width:420px;
  box-shadow:0 20px 60px rgba(0,0,0,.25);}
.logo{display:flex;align-items:center;gap:12px;margin-bottom:28px;}
.li{width:44px;height:44px;background:#2563eb;border-radius:12px;font-size:22px;
  display:flex;align-items:center;justify-content:center;}
h2{font-size:22px;font-weight:700;color:#0f2a4a;margin-bottom:4px;}
.sub{color:#64748b;font-size:13px;margin-bottom:24px;}
.fg{margin-bottom:14px;}
label{display:block;font-size:12px;font-weight:600;color:#334155;margin-bottom:5px;}
input{width:100%;padding:10px 13px;border:1.5px solid #e2e8f0;border-radius:8px;
  font-family:inherit;font-size:14px;outline:none;}
input:focus{border-color:#2563eb;}
.btn{width:100%;padding:12px;background:#2563eb;color:#fff;border:none;border-radius:9px;
  font-size:15px;font-weight:600;cursor:pointer;font-family:inherit;margin-top:6px;}
.lnk{text-align:center;margin-top:16px;font-size:14px;color:#64748b;}
.lnk a{color:#2563eb;text-decoration:none;font-weight:600;}
.err{background:#fee2e2;color:#991b1b;border:1px solid #fecaca;
  border-radius:8px;padding:10px;font-size:13px;margin-bottom:14px;}
</style></head><body><div class="box">
  <div class="logo"><div class="li">&#x1F3E5;</div>
  <span style="font-size:20px;font-weight:700;color:#0f2a4a;">MediSys</span></div>
  <h2>Create account</h2><p class="sub">Join MediSys to manage patient healthcare</p>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
  <form method="post">
    <div class="fg"><label>Full Name</label><input name="name" placeholder="Dr. Jane Smith" required></div>
    <div class="fg"><label>Email</label><input name="email" type="email" placeholder="you@hospital.com" required></div>
    <div class="fg"><label>Password</label><input name="password" type="password" placeholder="Min 6 chars" required></div>
    <button class="btn">Create Account</button>
  </form>
  <div class="lnk">Already registered? <a href="/login">Sign in</a></div>
</div></body></html>"""

@app.route("/")
def home():
    if "user" not in session: return redirect("/login")
    return redirect("/dashboard")

@app.route("/login", methods=["GET","POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form["email"].strip()
        pw = hash_pw(request.form["password"])
        conn = get_db()
        u = conn.execute("SELECT * FROM users WHERE email=? AND password=?", (email,pw)).fetchone()
        conn.close()
        if u:
            session["user"] = email
            session["uname"] = u["name"] or email.split("@")[0].title()
            return redirect("/dashboard")
        error = "Invalid email or password."
    return render_template_string(LOGIN_HTML, error=error)

@app.route("/signup", methods=["GET","POST"])
def signup():
    error = None
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip()
        pw = request.form["password"]
        if len(pw) < 6:
            error = "Password must be at least 6 characters."
        else:
            try:
                conn = get_db()
                conn.execute("INSERT INTO users(name,email,password) VALUES(?,?,?)",
                    (name,email,hash_pw(pw)))
                conn.commit(); conn.close()
                return redirect("/login")
            except:
                error = "Email already in use."
    return render_template_string(SIGNUP_HTML, error=error)

@app.route("/logout")
def logout():
    session.clear(); return redirect("/login")

@app.route("/dashboard")
def dashboard():
    if "user" not in session: return redirect("/login")
    conn = get_db()
    u = session["user"]
    total = conn.execute("SELECT COUNT(*) FROM patients WHERE created_by=?",(u,)).fetchone()[0]
    high  = conn.execute("SELECT COUNT(*) FROM patients WHERE created_by=? AND risk_level='High Risk'",(u,)).fetchone()[0]
    low   = conn.execute("SELECT COUNT(*) FROM patients WHERE created_by=? AND risk_level='Low Risk'",(u,)).fetchone()[0]
    appts = conn.execute("SELECT COUNT(*) FROM appointments WHERE created_by=? AND status='Scheduled'",(u,)).fetchone()[0]
    recent  = conn.execute("SELECT * FROM patients WHERE created_by=? ORDER BY id DESC LIMIT 5",(u,)).fetchall()
    upcoming= conn.execute("SELECT * FROM appointments WHERE created_by=? AND status='Scheduled' ORDER BY date,time LIMIT 4",(u,)).fetchall()
    conn.close()

    def rbadge(rl):
        if rl=="High Risk": return "<span class='badge badge-red'>High Risk</span>"
        if rl=="Low Risk":  return "<span class='badge badge-green'>Low Risk</span>"
        return "<span class='badge badge-blue'>N/A</span>"

    prows = "".join(f"<tr><td><b>{p['name']}</b></td><td>{p['age'] or '-'}</td>"
        f"<td>{p['gender'] or '-'}</td><td>{rbadge(p['risk_level'])}</td>"
        f"<td>{p['date_of_visit'] or '-'}</td>"
        f"<td><a href='/patients/{p['id']}' class='btn btn-outline btn-sm'>View</a></td></tr>"
        for p in recent) or "<tr><td colspan='6' style='text-align:center;padding:24px;color:#94a3b8'>No patients yet. <a href='/patients/add'>Add one</a></td></tr>"

    arows = "".join(f"<tr><td><b>{a['patient_name']}</b></td><td>{a['doctor']}</td>"
        f"<td>{a['date']} {a['time']}</td><td><span class='badge badge-blue'>{a['status']}</span></td></tr>"
        for a in upcoming) or "<tr><td colspan='4' style='text-align:center;padding:24px;color:#94a3b8'>No upcoming appointments.</td></tr>"

    uname = session.get("uname","Doctor")
    content = f"""
<div class="topbar">
  <div><h1>Dashboard</h1><p>Welcome back, {uname}. Here is your overview.</p></div>
  <div class="flex gap-2">
    <a href="/patients/add" class="btn btn-primary">+ Add Patient</a>
    <a href="/predict" class="btn btn-outline">Run Prediction</a>
  </div>
</div>
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-icon" style="background:#dbeafe;color:#2563eb"><i class="fa fa-users"></i></div>
    <div class="stat-value" style="color:#2563eb">{total}</div>
    <div class="stat-label">Total Patients</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon" style="background:#fee2e2;color:#ef4444"><i class="fa fa-triangle-exclamation"></i></div>
    <div class="stat-value" style="color:#ef4444">{high}</div>
    <div class="stat-label">High Risk</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon" style="background:#d1fae5;color:#10b981"><i class="fa fa-circle-check"></i></div>
    <div class="stat-value" style="color:#10b981">{low}</div>
    <div class="stat-label">Low Risk</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon" style="background:#fef3c7;color:#f59e0b"><i class="fa fa-calendar"></i></div>
    <div class="stat-value" style="color:#f59e0b">{appts}</div>
    <div class="stat-label">Upcoming Appointments</div>
  </div>
</div>
<div class="flex gap-3" style="align-items:flex-start">
  <div class="card" style="flex:2">
    <div class="section-title">Recent Patients</div>
    <table><thead><tr><th>Name</th><th>Age</th><th>Gender</th><th>Risk</th><th>Visit Date</th><th></th></tr></thead>
    <tbody>{prows}</tbody></table>
  </div>
  <div class="card" style="flex:1">
    <div class="section-title">Upcoming Appointments</div>
    <table><thead><tr><th>Patient</th><th>Doctor</th><th>When</th><th>Status</th></tr></thead>
    <tbody>{arows}</tbody></table>
  </div>
</div>"""
    return base(content,"dashboard")

@app.route("/patients")
def patients():
    if "user" not in session: return redirect("/login")
    q  = request.args.get("q","").strip()
    rf = request.args.get("risk","")
    conn = get_db()
    sql = "SELECT * FROM patients WHERE created_by=?"
    params = [session["user"]]
    if q:  sql += " AND name LIKE ?"; params.append(f"%{q}%")
    if rf: sql += " AND risk_level=?"; params.append(rf)
    sql += " ORDER BY id DESC"
    rows = conn.execute(sql,params).fetchall()
    conn.close()

    def rbadge(rl):
        if rl=="High Risk": return "<span class='badge badge-red'>High Risk</span>"
        if rl=="Low Risk":  return "<span class='badge badge-green'>Low Risk</span>"
        return "<span class='badge badge-blue'>N/A</span>"

    trows = "".join(f"""<tr>
      <td><b>{p['name']}</b></td><td>{p['age'] or '-'}</td><td>{p['gender'] or '-'}</td>
      <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{p['symptoms'] or '-'}</td>
      <td>{rbadge(p['risk_level'])}</td><td>{p['date_of_visit'] or '-'}</td>
      <td class="flex gap-2">
        <a href="/patients/{p['id']}" class="btn btn-outline btn-sm">View</a>
        <a href="/patients/{p['id']}/edit" class="btn btn-primary btn-sm">Edit</a>
        <a href="/patients/{p['id']}/delete" class="btn btn-danger btn-sm"
           onclick="return confirm('Delete this patient?')">Del</a>
      </td></tr>""" for p in rows) or \
      "<tr><td colspan='7' style='text-align:center;padding:32px;color:#94a3b8'>No patients found.</td></tr>"

    qv = q.replace('"',''); rfv = rf
    sel_h = "selected" if rfv=="High Risk" else ""
    sel_l = "selected" if rfv=="Low Risk" else ""
    content = f"""
<div class="topbar">
  <div><h1>Patient Records</h1><p>{len(rows)} patient(s)</p></div>
  <div class="flex gap-2">
    <a href="/patients/export" class="btn btn-outline">Export CSV</a>
    <a href="/patients/add" class="btn btn-primary">+ Add Patient</a>
  </div>
</div>
<div class="card">
  <form method="get" class="search-bar">
    <input name="q" placeholder="Search by name..." value="{qv}">
    <select name="risk">
      <option value="">All Risk Levels</option>
      <option {sel_h} value="High Risk">High Risk</option>
      <option {sel_l} value="Low Risk">Low Risk</option>
    </select>
    <button class="btn btn-primary" type="submit">Filter</button>
    <a href="/patients" class="btn btn-outline">Clear</a>
  </form>
  <table><thead><tr><th>Name</th><th>Age</th><th>Gender</th><th>Symptoms</th>
    <th>Risk</th><th>Visit Date</th><th>Actions</th></tr></thead>
  <tbody>{trows}</tbody></table>
</div>"""
    return base(content,"patients")

@app.route("/patients/add", methods=["GET","POST"])
def add_patient():
    if "user" not in session: return redirect("/login")
    error = None
    if request.method == "POST":
        name = request.form.get("name","").strip()
        if not name: error = "Patient name is required."
        else:
            conn = get_db()
            conn.execute("""INSERT INTO patients
              (name,age,gender,contact,symptoms,diagnosis,risk_level,date_of_visit,notes,created_by)
              VALUES(?,?,?,?,?,?,?,?,?,?)""", (
                name, request.form.get("age") or None,
                request.form.get("gender"), request.form.get("contact"),
                request.form.get("symptoms"), request.form.get("diagnosis"),
                request.form.get("risk_level","N/A"),
                request.form.get("date_of_visit") or datetime.now().strftime("%Y-%m-%d"),
                request.form.get("notes"), session["user"]
            ))
            conn.commit(); conn.close()
            return redirect("/patients")
    today = datetime.now().strftime("%Y-%m-%d")
    err_html = f'<div class="alert alert-danger">{error}</div>' if error else ""
    content = f"""
<div class="topbar"><div><h1>Add New Patient</h1><p>Fill in the details below</p></div></div>
<div class="card">{err_html}
  <form method="post"><div class="form-grid">
    <div class="form-group"><label>Full Name *</label><input name="name" placeholder="e.g. Amit Sharma" required></div>
    <div class="form-group"><label>Age</label><input name="age" type="number" min="0" max="120" placeholder="e.g. 45"></div>
    <div class="form-group"><label>Gender</label>
      <select name="gender"><option value="">-- Select --</option>
        <option>Male</option><option>Female</option><option>Other</option></select></div>
    <div class="form-group"><label>Contact Number</label><input name="contact" placeholder="+91 98765 43210"></div>
    <div class="form-group full"><label>Symptoms</label>
      <textarea name="symptoms" placeholder="Describe reported symptoms..."></textarea></div>
    <div class="form-group full"><label>Diagnosis</label>
      <textarea name="diagnosis" placeholder="Diagnosis or clinical notes..."></textarea></div>
    <div class="form-group"><label>Risk Level</label>
      <select name="risk_level"><option value="N/A">N/A</option>
        <option value="Low Risk">Low Risk</option><option value="High Risk">High Risk</option></select></div>
    <div class="form-group"><label>Date of Visit</label>
      <input name="date_of_visit" type="date" value="{today}"></div>
    <div class="form-group full"><label>Notes</label>
      <textarea name="notes" placeholder="Any additional notes..."></textarea></div>
  </div>
  <hr class="divider">
  <div class="flex gap-2 mt-4">
    <button class="btn btn-primary" type="submit">Save Patient</button>
    <a href="/patients" class="btn btn-outline">Cancel</a>
  </div></form>
</div>"""
    return base(content,"patients")

@app.route("/patients/<int:pid>")
def view_patient(pid):
    if "user" not in session: return redirect("/login")
    conn = get_db()
    p = conn.execute("SELECT * FROM patients WHERE id=? AND created_by=?",
        (pid,session["user"])).fetchone()
    conn.close()
    if not p: return redirect("/patients")
    def rbadge(rl):
        if rl=="High Risk": return "<span class='badge badge-red'>High Risk</span>"
        if rl=="Low Risk":  return "<span class='badge badge-green'>Low Risk</span>"
        return "<span class='badge badge-blue'>N/A</span>"
    content = f"""
<div class="topbar">
  <div><h1>{p['name']}</h1><p>Patient #{p['id']} - {p['date_of_visit'] or 'Unknown date'}</p></div>
  <div class="flex gap-2">
    <a href="/patients/{pid}/edit" class="btn btn-primary">Edit</a>
    <a href="/patients" class="btn btn-outline">Back</a>
  </div>
</div>
<div class="flex gap-3">
  <div class="card" style="flex:1">
    <div class="section-title">Patient Info</div>
    <table>
      <tr><td style="color:#64748b;width:130px">Name</td><td><b>{p['name']}</b></td></tr>
      <tr><td style="color:#64748b">Age</td><td>{p['age'] or '-'}</td></tr>
      <tr><td style="color:#64748b">Gender</td><td>{p['gender'] or '-'}</td></tr>
      <tr><td style="color:#64748b">Contact</td><td>{p['contact'] or '-'}</td></tr>
      <tr><td style="color:#64748b">Visit Date</td><td>{p['date_of_visit'] or '-'}</td></tr>
      <tr><td style="color:#64748b">Risk Level</td><td>{rbadge(p['risk_level'])}</td></tr>
    </table>
  </div>
  <div class="card" style="flex:2">
    <div class="section-title">Clinical Details</div>
    <div class="mb-4"><label style="font-size:12px;color:#64748b;font-weight:600;text-transform:uppercase">Symptoms</label>
      <p style="margin-top:6px;line-height:1.7">{p['symptoms'] or '-'}</p></div>
    <div class="mb-4"><label style="font-size:12px;color:#64748b;font-weight:600;text-transform:uppercase">Diagnosis</label>
      <p style="margin-top:6px;line-height:1.7">{p['diagnosis'] or '-'}</p></div>
    <div><label style="font-size:12px;color:#64748b;font-weight:600;text-transform:uppercase">Notes</label>
      <p style="margin-top:6px;line-height:1.7">{p['notes'] or '-'}</p></div>
  </div>
</div>"""
    return base(content,"patients")

@app.route("/patients/<int:pid>/edit", methods=["GET","POST"])
def edit_patient(pid):
    if "user" not in session: return redirect("/login")
    conn = get_db()
    p = conn.execute("SELECT * FROM patients WHERE id=? AND created_by=?",
        (pid,session["user"])).fetchone()
    if not p: conn.close(); return redirect("/patients")
    if request.method == "POST":
        conn.execute("""UPDATE patients SET name=?,age=?,gender=?,contact=?,symptoms=?,
          diagnosis=?,risk_level=?,date_of_visit=?,notes=? WHERE id=? AND created_by=?""", (
            request.form.get("name"), request.form.get("age") or None,
            request.form.get("gender"), request.form.get("contact"),
            request.form.get("symptoms"), request.form.get("diagnosis"),
            request.form.get("risk_level","N/A"), request.form.get("date_of_visit"),
            request.form.get("notes"), pid, session["user"]
        ))
        conn.commit(); conn.close()
        return redirect(f"/patients/{pid}")
    conn.close()
    def v(k): return p[k] or ""
    def sel(k,val): return "selected" if v(k)==val else ""
    content = f"""
<div class="topbar"><div><h1>Edit Patient</h1><p>{p['name']}</p></div></div>
<div class="card"><form method="post"><div class="form-grid">
    <div class="form-group"><label>Full Name *</label><input name="name" value="{v('name')}" required></div>
    <div class="form-group"><label>Age</label><input name="age" type="number" value="{v('age')}"></div>
    <div class="form-group"><label>Gender</label>
      <select name="gender"><option value="">--</option>
        <option {sel('gender','Male')}>Male</option>
        <option {sel('gender','Female')}>Female</option>
        <option {sel('gender','Other')}>Other</option></select></div>
    <div class="form-group"><label>Contact</label><input name="contact" value="{v('contact')}"></div>
    <div class="form-group full"><label>Symptoms</label><textarea name="symptoms">{v('symptoms')}</textarea></div>
    <div class="form-group full"><label>Diagnosis</label><textarea name="diagnosis">{v('diagnosis')}</textarea></div>
    <div class="form-group"><label>Risk Level</label>
      <select name="risk_level">
        <option {sel('risk_level','N/A')} value="N/A">N/A</option>
        <option {sel('risk_level','Low Risk')} value="Low Risk">Low Risk</option>
        <option {sel('risk_level','High Risk')} value="High Risk">High Risk</option></select></div>
    <div class="form-group"><label>Date of Visit</label>
      <input name="date_of_visit" type="date" value="{v('date_of_visit')}"></div>
    <div class="form-group full"><label>Notes</label><textarea name="notes">{v('notes')}</textarea></div>
</div>
<hr class="divider">
<div class="flex gap-2 mt-4">
  <button class="btn btn-primary" type="submit">Update Patient</button>
  <a href="/patients/{pid}" class="btn btn-outline">Cancel</a>
</div></form></div>"""
    return base(content,"patients")

@app.route("/patients/<int:pid>/delete")
def delete_patient(pid):
    if "user" not in session: return redirect("/login")
    conn = get_db()
    conn.execute("DELETE FROM patients WHERE id=? AND created_by=?",(pid,session["user"]))
    conn.commit(); conn.close()
    return redirect("/patients")

@app.route("/patients/export")
def export_patients():
    if "user" not in session: return redirect("/login")
    conn = get_db()
    rows = conn.execute("SELECT * FROM patients WHERE created_by=?",(session["user"],)).fetchall()
    conn.close()
    si = io.StringIO()
    wr = csv.writer(si)
    wr.writerow(["ID","Name","Age","Gender","Contact","Symptoms","Diagnosis","Risk Level","Date of Visit","Notes"])
    for r in rows:
        wr.writerow([r["id"],r["name"],r["age"],r["gender"],r["contact"],r["symptoms"],
                     r["diagnosis"],r["risk_level"],r["date_of_visit"],r["notes"]])
    out = make_response(si.getvalue())
    out.headers["Content-Disposition"] = "attachment; filename=patients.csv"
    out.headers["Content-type"] = "text/csv"
    return out

@app.route("/predict", methods=["GET","POST"])
def predict():
    if "user" not in session: return redirect("/login")
    risk = None; result_html = ""
    if request.method == "POST":
        try:
            vals = [float(request.form[k].strip()) for k,*_ in questions]
            pred = model.predict([vals])[0]
            if pred == 1:
                result_html = "<div class='alert alert-danger' style='font-size:16px;font-weight:600'>High Risk of Heart Disease — Please consult a cardiologist immediately.</div>"
                risk = "High Risk"
            else:
                result_html = "<div class='alert alert-success' style='font-size:16px;font-weight:600'>Low Risk of Heart Disease — Maintain healthy habits and schedule regular check-ups.</div>"
                risk = "Low Risk"
        except Exception as e:
            result_html = f"<div class='alert alert-danger'>Error: {e}</div>"

    fhtml = ""
    for k,lbl,ph in questions:
        val = request.form.get(k,"") if request.method=="POST" else ""
        fhtml += f"""<div class="form-group">
          <label>{lbl}</label>
          <input name="{k}" type="number" step="any" placeholder="{ph}" value="{val}" required>
        </div>"""

    content = f"""
<div class="topbar"><div><h1>Heart Disease Risk Prediction</h1>
  <p>Enter clinical values for an AI-powered risk assessment</p></div></div>
{result_html}
<div class="flex gap-3" style="align-items:flex-start">
  <div class="card" style="flex:2">
    <div class="section-title">Clinical Input Values</div>
    <form method="post">
      <div class="form-grid">{fhtml}</div>
      <hr class="divider">
      <button class="btn btn-primary mt-4" type="submit">Run Prediction</button>
    </form>
  </div>
  <div class="card" style="flex:1">
    <div class="section-title">Field Guide</div>
    <ul style="font-size:13px;color:#64748b;line-height:2.2;list-style:none">
      <li><b>cp</b>: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic</li>
      <li><b>restecg</b>: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy</li>
      <li><b>slope</b>: 0=upsloping, 1=flat, 2=downsloping</li>
      <li><b>thal</b>: 0=normal, 1=fixed, 2=reversible, 3=other</li>
      <li><b>ca</b>: vessels coloured by fluoroscopy</li>
    </ul>
    <hr class="divider">
    <p style="font-size:12px;color:#94a3b8">Model: RandomForest trained on UCI Heart Disease dataset (1025 records). Decision-support only, not a medical diagnosis.</p>
  </div>
</div>"""
    return base(content,"predict")

@app.route("/appointments", methods=["GET","POST"])
def appointments():
    if "user" not in session: return redirect("/login")
    if request.method == "POST":
        conn = get_db()
        conn.execute("INSERT INTO appointments(patient_name,doctor,date,time,reason,status,created_by) VALUES(?,?,?,?,?,?,?)",(
            request.form["patient_name"], request.form["doctor"],
            request.form["date"], request.form["time"],
            request.form.get("reason",""), "Scheduled", session["user"]
        ))
        conn.commit(); conn.close()
        return redirect("/appointments")
    conn = get_db()
    rows = conn.execute("SELECT * FROM appointments WHERE created_by=? ORDER BY date,time",(session["user"],)).fetchall()
    conn.close()

    def abadge(s):
        if s=="Scheduled": return "<span class='badge badge-blue'>Scheduled</span>"
        if s=="Completed":  return "<span class='badge badge-green'>Completed</span>"
        return "<span class='badge badge-red'>Cancelled</span>"

    arows = "".join(f"""<tr>
      <td><b>{a['patient_name']}</b></td><td>{a['doctor']}</td>
      <td>{a['date']}</td><td>{a['time']}</td>
      <td style="max-width:180px">{a['reason'] or '-'}</td>
      <td>{abadge(a['status'])}</td>
      <td><a href="/appointments/{a['id']}/cancel" class="btn btn-danger btn-sm"
        onclick="return confirm('Cancel?')">Cancel</a></td>
    </tr>""" for a in rows) or \
      "<tr><td colspan='7' style='text-align:center;padding:28px;color:#94a3b8'>No appointments scheduled.</td></tr>"

    content = f"""
<div class="topbar"><div><h1>Appointments</h1>
  <p>Schedule and manage patient appointments</p></div></div>
<div class="flex gap-3" style="align-items:flex-start">
  <div class="card" style="flex:2">
    <div class="section-title">All Appointments</div>
    <table><thead><tr><th>Patient</th><th>Doctor</th><th>Date</th>
      <th>Time</th><th>Reason</th><th>Status</th><th></th></tr></thead>
    <tbody>{arows}</tbody></table>
  </div>
  <div class="card" style="flex:1">
    <div class="section-title">New Appointment</div>
    <form method="post" style="display:flex;flex-direction:column;gap:12px">
      <div class="form-group"><label>Patient Name *</label>
        <input name="patient_name" placeholder="Patient name" required></div>
      <div class="form-group"><label>Doctor *</label>
        <input name="doctor" placeholder="Dr. ..." required></div>
      <div class="form-group"><label>Date *</label>
        <input name="date" type="date" required></div>
      <div class="form-group"><label>Time *</label>
        <input name="time" type="time" required></div>
      <div class="form-group"><label>Reason</label>
        <textarea name="reason" placeholder="Reason for visit..." style="min-height:60px"></textarea></div>
      <button class="btn btn-primary" type="submit">Schedule Appointment</button>
    </form>
  </div>
</div>"""
    return base(content,"appointments")

@app.route("/appointments/<int:aid>/cancel")
def cancel_appt(aid):
    if "user" not in session: return redirect("/login")
    conn = get_db()
    conn.execute("UPDATE appointments SET status='Cancelled' WHERE id=? AND created_by=?",(aid,session["user"]))
    conn.commit(); conn.close()
    return redirect("/appointments")

@app.route("/chatbot")
def chatbot():
    if "user" not in session: return redirect("/login")
    content = """
<div class="topbar"><div><h1>MediBot</h1>
  <p>Your health assistant - ask about symptoms, heart health, or how to use MediSys</p></div></div>
<div class="card" style="max-width:680px">
  <div class="chat-window" id="cw">
    <div class="chat-msg bot">Hi! I am <b>MediBot</b>, your MediSys health assistant.<br>
    Ask me about symptoms, heart health tips, how to use the app, and more!</div>
  </div>
  <div class="chat-input-row">
    <input id="ci" placeholder="Type your message..." onkeydown="if(event.key==='Enter')send()">
    <button class="btn btn-primary" onclick="send()">Send</button>
  </div>
  <p class="text-muted mt-4">Try: "What are heart disease symptoms?", "How do I add a patient?", "Tips for healthy diet?"</p>
</div>
<script>
function send(){
  var inp=document.getElementById('ci'), msg=inp.value.trim();
  if(!msg)return;
  var cw=document.getElementById('cw');
  cw.innerHTML+='<div class="chat-msg user">'+msg+'</div>';
  inp.value='';
  fetch('/chatbot/reply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({msg:msg})})
    .then(r=>r.json()).then(d=>{
      cw.innerHTML+='<div class="chat-msg bot">'+d.reply+'</div>';
      cw.scrollTop=cw.scrollHeight;
    });
  cw.scrollTop=cw.scrollHeight;
}
</script>"""
    return base(content,"chatbot")

@app.route("/chatbot/reply", methods=["POST"])
def chatbot_reply_api():
    if "user" not in session: return jsonify({"reply":"Please log in."})
    data = request.get_json()
    return jsonify({"reply": chatbot_reply(data.get("msg",""))})

@app.route("/reports")
def reports():
    if "user" not in session: return redirect("/login")
    conn = get_db()
    all_p = conn.execute("SELECT * FROM patients WHERE created_by=? ORDER BY id DESC",(session["user"],)).fetchall()
    conn.close()
    total = len(all_p)
    high = sum(1 for p in all_p if p["risk_level"]=="High Risk")
    low  = sum(1 for p in all_p if p["risk_level"]=="Low Risk")
    pct_h = round(high/total*100) if total else 0
    pct_l = round(low/total*100) if total else 0

    def rbadge(rl):
        if rl=="High Risk": return "<span class='badge badge-red'>High Risk</span>"
        if rl=="Low Risk":  return "<span class='badge badge-green'>Low Risk</span>"
        return "<span class='badge badge-blue'>N/A</span>"

    prows = "".join(f"""<tr>
      <td>#{p['id']}</td><td><b>{p['name']}</b></td><td>{p['age'] or '-'}</td>
      <td>{p['gender'] or '-'}</td><td>{rbadge(p['risk_level'])}</td>
      <td>{p['date_of_visit'] or '-'}</td>
      <td style="max-width:200px;font-size:13px;color:#64748b">{(p['diagnosis'] or '-')[:60]}</td>
    </tr>""" for p in all_p) or \
      "<tr><td colspan='7' style='text-align:center;padding:28px;color:#94a3b8'>No patient data.</td></tr>"

    content = f"""
<div class="topbar">
  <div><h1>Reports &amp; Summary</h1><p>Overview of all patient data</p></div>
  <a href="/patients/export" class="btn btn-primary">Export CSV</a>
</div>
<div class="stat-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:24px">
  <div class="stat-card">
    <div class="stat-icon" style="background:#dbeafe;color:#2563eb"><i class="fa fa-users"></i></div>
    <div class="stat-value" style="color:#2563eb">{total}</div>
    <div class="stat-label">Total Patients</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon" style="background:#fee2e2;color:#ef4444"><i class="fa fa-heart-crack"></i></div>
    <div class="stat-value" style="color:#ef4444">{high} <small style="font-size:14px;font-weight:400">({pct_h}%)</small></div>
    <div class="stat-label">High Risk</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon" style="background:#d1fae5;color:#10b981"><i class="fa fa-heart"></i></div>
    <div class="stat-value" style="color:#10b981">{low} <small style="font-size:14px;font-weight:400">({pct_l}%)</small></div>
    <div class="stat-label">Low Risk</div>
  </div>
</div>
<div class="card">
  <div class="section-title">Full Patient Report</div>
  <table><thead><tr><th>ID</th><th>Name</th><th>Age</th><th>Gender</th>
    <th>Risk</th><th>Visit Date</th><th>Diagnosis Summary</th></tr></thead>
  <tbody>{prows}</tbody></table>
</div>"""
    return base(content,"reports")

@app.route("/chat")
def chat_legacy():
    return redirect("/predict")

if __name__ == "__main__":
    app.run(debug=True)
