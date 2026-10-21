import os
import pickle
import warnings

import numpy as np
from flask import Flask, jsonify, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model (path is relative to this file so it works on Vercel too)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "logistic.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Model input order:
# Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
# CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
#
# Categorical values were label-encoded alphabetically during training:
GENDER_MAP = {"female": 0, "male": 1}
MARRIED_MAP = {"no": 0, "yes": 1}
EDUCATION_MAP = {"graduate": 0, "not_graduate": 1}
SELF_EMP_MAP = {"no": 0, "yes": 1}
AREA_MAP = {"rural": 0, "semiurban": 1, "urban": 2}
AREA_NAME = {"rural": "rural", "semiurban": "semi-urban", "urban": "urban"}
DEPENDENTS_ALLOWED = {"0", "1", "2", "3"}  # "3" means 3 or more

# Index of the class "approved" (1) in the model output
APPROVED_IDX = list(model.classes_).index(1)


def indian_format(n: float) -> str:
    """Format a number with Indian digit grouping, e.g. 4,65,56,100."""
    s = str(int(round(n)))
    if len(s) <= 3:
        return s
    last3, rest = s[-3:], s[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return ",".join(groups + [last3])


def money(x: float) -> str:
    return "\u20b9" + indian_format(x)


def money_words(x: float) -> str:
    """Easy-to-read form: '₹3.88 Crore' / '₹45.20 Lakh'."""
    if x >= 1e7:
        return f"\u20b9{x / 1e7:.2f} Crore"
    if x >= 1e5:
        return f"\u20b9{x / 1e5:.2f} Lakh"
    return money(x)


def term_text(months: int) -> str:
    if months % 12 == 0:
        y = months // 12
        return f"{y} year{'s' if y != 1 else ''}"
    return f"{months} months"


def approval_probability(row) -> float:
    x = np.array([row], dtype=float)
    return float(model.predict_proba(x)[0][APPROVED_IDX])


# ---------------------------------------------------------------------------
# Frontend (single page, light + dark theme)
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loan Approval Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #f4f7fb;
    --bg-accent: radial-gradient(1200px 500px at 8% -10%, #d9e8ff 0%, transparent 60%),
                 radial-gradient(900px 500px at 100% 0%, #d8f5e8 0%, transparent 55%);
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --input-bg: #f8fafc;
    --primary: #2563eb;
    --primary-2: #06b6d4;
    --primary-soft: #eaf1ff;
    --good: #059669;
    --good-soft: #e7f8f0;
    --mid: #d97706;
    --mid-soft: #fff6e5;
    --bad: #dc2626;
    --bad-soft: #fdecec;
    --track: #e2e8f0;
    --shadow: 0 10px 30px rgba(15, 23, 42, .08);
  }
  :root[data-theme="dark"] {
    --bg: #0a1120;
    --bg-accent: radial-gradient(1200px 500px at 8% -10%, #10264f 0%, transparent 60%),
                 radial-gradient(900px 500px at 100% 0%, #0d2d2a 0%, transparent 55%);
    --card: #101a2e;
    --text: #e7edf8;
    --muted: #8fa1c0;
    --border: #1f2d4a;
    --input-bg: #0b1424;
    --primary: #60a5fa;
    --primary-2: #22d3ee;
    --primary-soft: #14284a;
    --good: #34d399;
    --good-soft: #0f2d25;
    --mid: #fbbf24;
    --mid-soft: #2e2510;
    --bad: #f87171;
    --bad-soft: #2f1719;
    --track: #1f2d4a;
    --shadow: 0 10px 30px rgba(0, 0, 0, .5);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); background-image: var(--bg-accent); background-attachment: fixed;
    color: var(--text); min-height: 100vh; line-height: 1.5;
    transition: background .25s, color .25s;
  }
  .wrap { max-width: 1040px; margin: 0 auto; padding: 28px 18px 60px; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 26px; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo {
    width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
    background: linear-gradient(135deg, var(--primary), var(--primary-2)); color: #fff;
  }
  .brand h1 { font-size: 1.15rem; margin: 0; letter-spacing: -.01em; }
  .brand p { margin: 0; font-size: .8rem; color: var(--muted); }
  .theme-btn {
    border: 1px solid var(--border); background: var(--card); color: var(--text);
    width: 42px; height: 42px; border-radius: 12px; cursor: pointer; display: grid; place-items: center;
    box-shadow: var(--shadow); transition: transform .15s;
  }
  .theme-btn:hover { transform: translateY(-1px); }
  .grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 20px; align-items: start; }
  @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 18px;
    padding: 24px; box-shadow: var(--shadow); transition: background .25s, border-color .25s;
  }
  .card h2 { margin: 0 0 4px; font-size: 1.05rem; }
  .card .sub { margin: 0 0 8px; color: var(--muted); font-size: .88rem; }
  .section-title {
    display: flex; align-items: center; gap: 8px; margin: 22px 0 12px; font-size: .78rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; color: var(--primary);
  }
  .section-title::after { content: ""; flex: 1; height: 1px; background: var(--border); }
  .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 540px) { .fields { grid-template-columns: 1fr; } }
  .field.full { grid-column: 1 / -1; }
  .lbl { display: block; font-size: .82rem; font-weight: 600; margin-bottom: 6px; }
  .lbl small { color: var(--muted); font-weight: 500; }
  .hint { font-size: .74rem; color: var(--muted); margin-top: 5px; }
  input[type=number], select {
    width: 100%; padding: 11px 13px; font: inherit; color: var(--text);
    background: var(--input-bg); border: 1px solid var(--border); border-radius: 10px; outline: none;
    transition: border-color .15s, box-shadow .15s;
  }
  input[type=number]:focus, select:focus { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-soft); }
  .seg { display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; gap: 6px; padding: 4px; background: var(--input-bg); border: 1px solid var(--border); border-radius: 12px; }
  .seg input { position: absolute; opacity: 0; pointer-events: none; }
  .seg label { text-align: center; padding: 8px 6px; border-radius: 9px; font-size: .86rem; font-weight: 600; cursor: pointer; color: var(--muted); transition: all .15s; white-space: nowrap; }
  .seg input:checked + label { background: linear-gradient(135deg, var(--primary), var(--primary-2)); color: #fff; box-shadow: 0 4px 12px rgba(37,99,235,.28); }
  .seg input:focus-visible + label { outline: 2px solid var(--primary); outline-offset: 2px; }
  button.primary {
    margin-top: 26px; width: 100%; padding: 14px 16px; font: inherit; font-weight: 600; color: #fff;
    background: linear-gradient(135deg, var(--primary), var(--primary-2)); border: 0; border-radius: 12px; cursor: pointer;
    transition: transform .15s, opacity .15s; display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  button.primary:hover { transform: translateY(-1px); }
  button.primary:disabled { opacity: .65; cursor: wait; transform: none; }
  .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,.4); border-top-color: #fff; border-radius: 50%; animation: spin .7s linear infinite; display: none; }
  button.loading .spinner { display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .placeholder { text-align: center; padding: 40px 10px; color: var(--muted); }
  .placeholder svg { opacity: .55; margin-bottom: 10px; }
  .result { display: none; animation: fade .35s ease; }
  @keyframes fade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
  .gauge-wrap { display: flex; flex-direction: column; align-items: center; margin: 6px 0 10px; }
  .gauge { position: relative; width: 170px; height: 170px; }
  .gauge svg { transform: rotate(-90deg); }
  .gauge .track { stroke: var(--track); }
  .gauge .bar { transition: stroke-dashoffset .9s cubic-bezier(.2,.8,.2,1), stroke .3s; }
  .gauge .center { position: absolute; inset: 0; display: grid; place-content: center; text-align: center; }
  .gauge .pct { font-size: 2.3rem; font-weight: 700; letter-spacing: -.03em; line-height: 1; }
  .gauge .cap { font-size: .72rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
  .verdict { margin-top: 12px; display: inline-flex; align-items: center; gap: 8px; padding: 7px 16px; border-radius: 999px; font-weight: 700; font-size: .95rem; }
  .tone-good { color: var(--good); } .tone-good .bar { stroke: var(--good); } .verdict.tone-good { background: var(--good-soft); }
  .tone-mid  { color: var(--mid); }  .tone-mid .bar  { stroke: var(--mid); }  .verdict.tone-mid  { background: var(--mid-soft); }
  .tone-bad  { color: var(--bad); }  .tone-bad .bar  { stroke: var(--bad); }  .verdict.tone-bad  { background: var(--bad-soft); }
  .summary { margin: 16px 0 12px; padding: 14px 16px; border-radius: 12px; background: var(--primary-soft); font-size: .92rem; }
  .insight { margin: 0 0 14px; padding: 12px 14px; border-radius: 12px; background: var(--mid-soft); color: var(--mid); font-size: .88rem; display: none; }
  .factors { list-style: none; padding: 0; margin: 0 0 14px; display: grid; gap: 8px; }
  .factors li { display: flex; gap: 10px; align-items: flex-start; font-size: .88rem; padding: 10px 12px; border: 1px solid var(--border); border-radius: 12px; }
  .dot { flex: none; width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center; font-size: .72rem; font-weight: 700; color: #fff; margin-top: 1px; }
  .dot.good { background: var(--good); } .dot.bad { background: var(--bad); } .dot.info { background: var(--primary); }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .stat { border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; }
  .stat b { display: block; font-size: 1rem; }
  .stat span { font-size: .75rem; color: var(--muted); }
  .error { display: none; margin-top: 16px; padding: 12px 14px; border-radius: 10px; background: var(--bad-soft); color: var(--bad); font-size: .88rem; }
  .note { margin-top: 16px; font-size: .75rem; color: var(--muted); }
  footer { text-align: center; color: var(--muted); font-size: .78rem; margin-top: 30px; }
</style>
<script>
  // Apply saved theme (or system preference) before paint to avoid flashing
  (function () {
    var saved = null;
    try { saved = localStorage.getItem("theme"); } catch (e) {}
    var dark = saved ? saved === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  })();
</script>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21h18"/><path d="M5 21V10l7-6 7 6v11"/><path d="M9 21v-6h6v6"/></svg>
      </div>
      <div>
        <h1>Loan Approval Predictor</h1>
        <p>Check your chances before you apply</p>
      </div>
    </div>
    <button class="theme-btn" id="themeBtn" aria-label="Toggle theme" title="Toggle theme"></button>
  </header>

  <div class="grid">
    <form class="card" id="form" novalidate>
      <h2>Applicant details</h2>
      <p class="sub">Fill in the form below. It only takes a minute.</p>

      <div class="section-title">Personal</div>
      <div class="fields">
        <div class="field">
          <span class="lbl">Gender</span>
          <div class="seg">
            <input type="radio" name="gender" id="gM" value="male" checked><label for="gM">Male</label>
            <input type="radio" name="gender" id="gF" value="female"><label for="gF">Female</label>
          </div>
        </div>
        <div class="field">
          <span class="lbl">Married</span>
          <div class="seg">
            <input type="radio" name="married" id="mY" value="yes" checked><label for="mY">Yes</label>
            <input type="radio" name="married" id="mN" value="no"><label for="mN">No</label>
          </div>
        </div>
        <div class="field">
          <label class="lbl" for="dependents">Dependents</label>
          <select id="dependents" name="dependents">
            <option value="0" selected>0</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3 or more</option>
          </select>
        </div>
        <div class="field">
          <label class="lbl" for="education">Education</label>
          <select id="education" name="education">
            <option value="graduate" selected>Graduate</option>
            <option value="not_graduate">Not graduate</option>
          </select>
        </div>
        <div class="field full">
          <span class="lbl">Self-employed</span>
          <div class="seg">
            <input type="radio" name="self_employed" id="sN" value="no" checked><label for="sN">No</label>
            <input type="radio" name="self_employed" id="sY" value="yes"><label for="sY">Yes</label>
          </div>
        </div>
      </div>

      <div class="section-title">Income</div>
      <div class="fields">
        <div class="field">
          <label class="lbl" for="applicant_income">Your monthly income <small>(₹)</small></label>
          <input type="number" id="applicant_income" name="applicant_income" min="0" max="1000000" step="1" value="5000" required>
        </div>
        <div class="field">
          <label class="lbl" for="coapplicant_income">Co-applicant monthly income <small>(₹)</small></label>
          <input type="number" id="coapplicant_income" name="coapplicant_income" min="0" max="1000000" step="1" value="0" required>
          <div class="hint">Enter 0 if there is no co-applicant.</div>
        </div>
      </div>

      <div class="section-title">Loan &amp; property</div>
      <div class="fields">
        <div class="field">
          <label class="lbl" for="loan_amount">Loan amount <small>(in ₹ thousands)</small></label>
          <input type="number" id="loan_amount" name="loan_amount" min="1" max="2000" step="1" value="130" required>
          <div class="hint" id="loanHint">130 = ₹1,30,000</div>
        </div>
        <div class="field">
          <label class="lbl" for="loan_term">Loan term <small>(months)</small></label>
          <input type="number" id="loan_term" name="loan_term" min="12" max="480" step="12" value="360" required>
          <div class="hint" id="termHint">360 months = 30 years</div>
        </div>
        <div class="field">
          <label class="lbl" for="property_area">Property area</label>
          <select id="property_area" name="property_area">
            <option value="urban">Urban</option>
            <option value="semiurban" selected>Semi-urban</option>
            <option value="rural">Rural</option>
          </select>
        </div>
        <div class="field">
          <span class="lbl">Credit history</span>
          <div class="seg">
            <input type="radio" name="credit_history" id="cG" value="1" checked><label for="cG">Good</label>
            <input type="radio" name="credit_history" id="cP" value="0"><label for="cP">Poor / none</label>
          </div>
          <div class="hint">Good = past loans/cards repaid on time.</div>
        </div>
      </div>

      <button class="primary" id="submitBtn" type="submit">
        <span class="spinner"></span><span id="btnText">Check approval chance</span>
      </button>
      <div class="error" id="error"></div>
    </form>

    <div class="card">
      <h2>Your result</h2>
      <p class="sub">The prediction will appear here.</p>

      <div class="placeholder" id="placeholder">
        <svg width="54" height="54" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/></svg>
        <div>Complete the form and press <b>Check approval chance</b>.</div>
      </div>

      <div class="result" id="result">
        <div class="gauge-wrap" id="gaugeWrap">
          <div class="gauge">
            <svg width="170" height="170" viewBox="0 0 170 170">
              <circle class="track" cx="85" cy="85" r="72" fill="none" stroke-width="14"/>
              <circle class="bar" id="bar" cx="85" cy="85" r="72" fill="none" stroke-width="14" stroke-linecap="round" stroke-dasharray="452.4" stroke-dashoffset="452.4"/>
            </svg>
            <div class="center"><div class="pct" id="pct">0%</div><div class="cap">approval chance</div></div>
          </div>
          <div class="verdict" id="verdict"></div>
        </div>
        <div class="summary" id="summary"></div>
        <div class="insight" id="insight"></div>
        <ul class="factors" id="factors"></ul>
        <div class="stats">
          <div class="stat"><b id="income"></b><span>Total monthly income</span></div>
          <div class="stat"><b id="installment"></b><span>Loan repayment / month (before interest)</span></div>
        </div>
        <div class="note">This is an automated estimate from a statistical model, not a lending decision. The actual outcome depends on the lender's own checks and policies.</div>
      </div>
    </div>
  </div>

  <footer>Built with Flask &amp; scikit-learn</footer>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  const CIRC = 2 * Math.PI * 72;
  const sun = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  const moon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';

  function paintThemeIcon() {
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    $("themeBtn").innerHTML = dark ? sun : moon;
  }
  paintThemeIcon();
  $("themeBtn").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
    paintThemeIcon();
  });

  // Indian digit grouping for hints
  function inr(n) {
    n = Math.round(n); let s = String(n);
    if (s.length <= 3) return "\u20b9" + s;
    let last3 = s.slice(-3), rest = s.slice(0, -3), g = [];
    while (rest.length > 2) { g.unshift(rest.slice(-2)); rest = rest.slice(0, -2); }
    if (rest) g.unshift(rest);
    return "\u20b9" + g.concat(last3).join(",");
  }
  function updateHints() {
    const la = parseFloat($("loan_amount").value), lt = parseInt($("loan_term").value, 10);
    $("loanHint").textContent = isFinite(la) ? la + " = " + inr(la * 1000) : "";
    if (isFinite(lt) && lt > 0) {
      $("termHint").textContent = lt + " months" + (lt % 12 === 0 ? " = " + (lt / 12) + " year" + (lt / 12 !== 1 ? "s" : "") : "");
    } else { $("termHint").textContent = ""; }
  }
  $("loan_amount").addEventListener("input", updateHints);
  $("loan_term").addEventListener("input", updateHints);
  updateHints();

  function showResult(d) {
    const tone = "tone-" + d.tone;
    const wrap = $("gaugeWrap");
    wrap.className = "gauge-wrap " + tone;
    $("verdict").className = "verdict " + tone;
    $("verdict").textContent = d.verdict;
    $("pct").textContent = d.percent + "%";
    $("summary").textContent = d.summary;
    $("income").textContent = d.total_income;
    $("installment").textContent = d.installment;

    const ins = $("insight");
    if (d.insight) { ins.textContent = d.insight; ins.style.display = "block"; } else { ins.style.display = "none"; }

    const ul = $("factors"); ul.innerHTML = "";
    d.factors.forEach((f) => {
      const li = document.createElement("li");
      const dot = document.createElement("span");
      dot.className = "dot " + f.tone;
      dot.textContent = f.tone === "good" ? "\u2713" : (f.tone === "bad" ? "!" : "i");
      const t = document.createElement("span"); t.textContent = f.text;
      li.appendChild(dot); li.appendChild(t); ul.appendChild(li);
    });

    $("placeholder").style.display = "none";
    const r = $("result"); r.style.display = "block";
    r.style.animation = "none"; r.offsetHeight; r.style.animation = "";

    const bar = $("bar");
    bar.style.transition = "none"; bar.style.strokeDashoffset = CIRC; bar.getBoundingClientRect();
    bar.style.transition = "";
    requestAnimationFrame(() => { bar.style.strokeDashoffset = CIRC * (1 - d.percent / 100); });
    if (window.innerWidth <= 860) r.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = $("error"); err.style.display = "none";
    const btn = $("submitBtn");
    btn.disabled = true; btn.classList.add("loading"); $("btnText").textContent = "Checking...";

    const payload = {};
    new FormData($("form")).forEach((v, k) => payload[k] = v);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Something went wrong.");
      showResult(data);
    } catch (ex) {
      err.textContent = ex.message; err.style.display = "block";
    } finally {
      btn.disabled = false; btn.classList.remove("loading"); $("btnText").textContent = "Check approval chance";
    }
  });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return HTML


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()

    def number(key, label, lo, hi):
        try:
            val = float(data.get(key))
        except (TypeError, ValueError):
            raise ValueError(f"Please enter a valid number for '{label}'.")
        if not (lo <= val <= hi):
            raise ValueError(f"'{label}' must be between {lo:g} and {hi:g}.")
        return val

    try:
        app_income = number("applicant_income", "Your monthly income", 0, 1_000_000)
        co_income = number("coapplicant_income", "Co-applicant income", 0, 1_000_000)
        loan_amount = number("loan_amount", "Loan amount", 1, 2000)
        loan_term = number("loan_term", "Loan term", 12, 480)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400

    gender = str(data.get("gender", "")).lower()
    married = str(data.get("married", "")).lower()
    education = str(data.get("education", "")).lower()
    self_emp = str(data.get("self_employed", "")).lower()
    area = str(data.get("property_area", "")).lower()
    dependents = str(data.get("dependents", "")).strip()
    credit = str(data.get("credit_history", "")).strip()

    if (
        gender not in GENDER_MAP
        or married not in MARRIED_MAP
        or education not in EDUCATION_MAP
        or self_emp not in SELF_EMP_MAP
        or area not in AREA_MAP
        or dependents not in DEPENDENTS_ALLOWED
        or credit not in {"0", "1"}
    ):
        return jsonify(ok=False, error="Please check that every option in the form is selected."), 400

    def build_row(credit_value):
        return [
            GENDER_MAP[gender],
            MARRIED_MAP[married],
            int(dependents),
            EDUCATION_MAP[education],
            SELF_EMP_MAP[self_emp],
            app_income,
            co_income,
            loan_amount,
            loan_term,
            int(credit_value),
            AREA_MAP[area],
        ]

    prob = approval_probability(build_row(credit))
    percent = int(round(prob * 100))

    if prob >= 0.75:
        verdict, tone = "Strong chance of approval", "good"
    elif prob >= 0.50:
        verdict, tone = "Likely to be approved", "good"
    elif prob >= 0.35:
        verdict, tone = "Approval is uncertain", "mid"
    else:
        verdict, tone = "Unlikely to be approved", "bad"

    # ---- plain-language summary -------------------------------------------
    total_income = app_income + co_income
    loan_rupees = loan_amount * 1000
    months = int(loan_term)
    installment = loan_rupees / months

    pct_article = "an" if percent in (8, 11, 18) or 80 <= percent <= 89 else "a"
    dep_txt = {"0": "no dependents", "1": "1 dependent", "2": "2 dependents", "3": "3 or more dependents"}[dependents]
    summary = (
        f"For a {'married' if married == 'yes' else 'single'} {gender} "
        f"{'graduate' if education == 'graduate' else 'non-graduate'} applicant "
        f"({'self-employed' if self_emp == 'yes' else 'salaried / not self-employed'}) with {dep_txt}, "
        f"a total household income of {money(total_income)} per month, applying for "
        f"{money_words(loan_rupees)} over {term_text(months)} on a property in a "
        f"{AREA_NAME[area]} area with {'a good' if credit == '1' else 'a poor or no'} credit history, "
        f"the model estimates {pct_article} {percent}% chance of approval."
    )

    # ---- what-if insight on credit history --------------------------------
    insight = ""
    if credit == "0":
        alt = approval_probability(build_row(1))
        if alt - prob > 0.02:
            insight = (
                f"Credit history matters most. With a good credit history, your estimated "
                f"chance would rise from {percent}% to about {int(round(alt * 100))}%."
            )
    else:
        alt = approval_probability(build_row(0))
        if prob - alt > 0.02:
            insight = (
                f"Your good credit history is helping a lot. Without it, the estimated "
                f"chance would drop to about {int(round(alt * 100))}%."
            )

    # ---- key factors (simple, readable hints) -----------------------------
    factors = []
    if credit == "1":
        factors.append({"tone": "good", "text": "Good credit history is the strongest positive signal for lenders."})
    else:
        factors.append({"tone": "bad", "text": "A poor or missing credit history is the biggest reason for rejection. Building a clean repayment record can help."})

    if total_income > 0:
        burden = installment / total_income
        if burden <= 0.30:
            factors.append({"tone": "good", "text": f"The monthly repayment ({money(installment)}) is a small part of your income, which is comfortable."})
        elif burden <= 0.50:
            factors.append({"tone": "info", "text": f"The monthly repayment ({money(installment)}) is a moderate share of your income."})
        else:
            factors.append({"tone": "bad", "text": f"The monthly repayment ({money(installment)}) is a large share of your income. A smaller loan or longer term may help."})
    else:
        factors.append({"tone": "bad", "text": "No income was entered, which makes approval very unlikely."})

    if co_income > 0:
        factors.append({"tone": "good", "text": "Having a co-applicant with income strengthens the application."})
    elif prob < 0.6:
        factors.append({"tone": "info", "text": "Adding a co-applicant with income could improve the application."})

    return jsonify(
        ok=True,
        percent=percent,
        verdict=verdict,
        tone=tone,
        summary=summary,
        insight=insight,
        factors=factors,
        total_income=money(total_income),
        installment=money(installment),
    )


if __name__ == "__main__":
    app.run(debug=True)
