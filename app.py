import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "logistic.pkl"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("loan-app")

app = Flask(__name__)
# AWS Elastic Beanstalk looks for a WSGI callable named "application" by default.
application = app

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
with open(MODEL_PATH, "rb") as fh:
    model = pickle.load(fh)

FEATURE_ORDER = list(
    getattr(
        model,
        "feature_names_in_",
        [
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area",
        ],
    )
)
log.info("Model loaded from %s with features: %s", MODEL_PATH, FEATURE_ORDER)

# --------------------------------------------------------------------------- #
# Input encoding
# IMPORTANT: these numeric codes must match how the data was encoded when the
# model was trained. If your notebook used different codes, edit them here.
# --------------------------------------------------------------------------- #
CATEGORICAL = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Credit_History": {"1": 1.0, "0": 0.0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}

# field -> (minimum, maximum, label used in error messages)
NUMERIC = {
    "ApplicantIncome": (0, 1_000_000, "Applicant income"),
    "CoapplicantIncome": (0, 1_000_000, "Co-applicant income"),
    "LoanAmount": (1, 10_000, "Loan amount"),
    "Loan_Amount_Term": (12, 480, "Loan term"),
}


def parse_payload(payload):
    """Validate raw input and return (feature_row, errors)."""
    row, errors = {}, {}

    for field, mapping in CATEGORICAL.items():
        value = str(payload.get(field, "")).strip()
        if value in mapping:
            row[field] = mapping[value]
        else:
            errors[field] = "Choose an option."

    for field, (low, high, label) in NUMERIC.items():
        raw = payload.get(field)
        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors[field] = f"{label} must be a number."
            continue
        if not low <= number <= high:
            errors[field] = f"{label} must be between {low:,} and {high:,}."
        else:
            row[field] = number

    return row, errors


# --------------------------------------------------------------------------- #
# Web page (HTML, CSS and JS in one place so the project needs only app.py)
# --------------------------------------------------------------------------- #
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Loan eligibility check</title>
  <meta name="description" content="Enter your application details and get an instant estimate of your loan approval chances.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Public+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --mist: #e8edf1;
      --paper: #ffffff;
      --ink: #11202f;
      --slate: #53636f;
      --rule: #cbd5dd;
      --cobalt: #2147c9;
      --cobalt-dark: #183597;
      --pine: #0f7a52;
      --pine-tint: #e3f3ec;
      --brick: #b23a2e;
      --brick-tint: #f9e8e5;
      --radius: 10px;
      --head: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
      --body: "Public Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
    }
    *, *::before, *::after { box-sizing: border-box; }
    html { -webkit-text-size-adjust: 100%; }
    body {
      margin: 0;
      background: var(--mist);
      color: var(--ink);
      font-family: var(--body);
      font-size: 1rem;
      line-height: 1.55;
    }

    .page { width: min(1120px, 100% - 2rem); margin-inline: auto; padding-block: 2.5rem 4rem; }

    header.intro { max-width: 40rem; margin-bottom: 2.25rem; }
    h1 {
      font-family: var(--head);
      font-weight: 700;
      font-size: clamp(2rem, 4.5vw, 3.1rem);
      line-height: 1.08;
      letter-spacing: -0.02em;
      margin: 0 0 0.75rem;
    }
    header.intro p { margin: 0; color: var(--slate); font-size: 1.075rem; }

    .layout { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr); gap: 2.5rem; align-items: start; }

    /* ---------- form ---------- */
    form { background: var(--paper); border-radius: var(--radius); padding: 0.5rem 1.75rem 1.75rem; }
    fieldset { border: 0; margin: 0; padding: 1.5rem 0; min-width: 0; }
    fieldset + fieldset { border-top: 1px solid var(--rule); }
    legend {
      font-family: var(--head);
      font-weight: 700;
      font-size: 1.2rem;
      letter-spacing: -0.01em;
      padding: 0;
      margin-bottom: 1rem;
    }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.1rem 1.25rem; }
    .field { display: flex; flex-direction: column; gap: 0.35rem; min-width: 0; }
    .field.wide { grid-column: 1 / -1; }
    .field > label, .field > .label { font-weight: 600; font-size: 0.925rem; }
    .hint { color: var(--slate); font-size: 0.825rem; margin: 0; }
    .error { color: var(--brick); font-size: 0.85rem; font-weight: 500; margin: 0; min-height: 0; }

    input[type="number"], select {
      width: 100%;
      font: inherit;
      color: var(--ink);
      background: var(--paper);
      border: 1.5px solid var(--rule);
      border-radius: 8px;
      padding: 0.65rem 0.8rem;
      min-height: 2.9rem;
      transition: border-color 0.15s;
    }
    input[type="number"]:hover, select:hover { border-color: #9fb0bd; }
    .field.invalid input, .field.invalid select { border-color: var(--brick); }

    :where(input, select, button, .seg label):focus-visible,
    .seg input:focus-visible + label {
      outline: 3px solid var(--cobalt);
      outline-offset: 2px;
    }

    .prefix { position: relative; }
    .prefix span {
      position: absolute; left: 0.8rem; top: 50%; transform: translateY(-50%);
      color: var(--slate); pointer-events: none;
    }
    .prefix input { padding-left: 1.8rem; }

    /* segmented radio controls */
    .seg { display: flex; border: 1.5px solid var(--rule); border-radius: 8px; overflow: hidden; }
    .seg input { position: absolute; opacity: 0; pointer-events: none; }
    .seg label {
      flex: 1; text-align: center; padding: 0.65rem 0.5rem; min-height: 2.9rem;
      display: flex; align-items: center; justify-content: center;
      cursor: pointer; font-size: 0.95rem; line-height: 1.2;
      border-left: 1.5px solid var(--rule);
      transition: background 0.15s, color 0.15s;
    }
    .seg label:first-of-type { border-left: 0; }
    .seg label:hover { background: var(--mist); }
    .seg input:checked + label { background: var(--cobalt); color: #fff; font-weight: 600; }
    .field.invalid .seg { border-color: var(--brick); }

    .actions { display: flex; gap: 0.75rem; align-items: center; padding-top: 1.5rem; border-top: 1px solid var(--rule); }
    button {
      font: inherit; font-weight: 600; cursor: pointer; border-radius: 8px;
      min-height: 3rem; padding: 0.7rem 1.5rem; border: 1.5px solid transparent;
    }
    .primary { background: var(--cobalt); color: #fff; }
    .primary:hover { background: var(--cobalt-dark); }
    .primary[disabled] { opacity: 0.65; cursor: progress; }
    .ghost { background: transparent; color: var(--ink); border-color: var(--rule); }
    .ghost:hover { background: var(--mist); }

    /* ---------- decision slip ---------- */
    .slip-wrap { position: sticky; top: 1.5rem; }
    .slip {
      background: var(--paper);
      border-radius: var(--radius);
      position: relative;
      overflow: hidden;
    }
    .slip-top { padding: 1.5rem 1.75rem 1.5rem; }
    .slip-top h2 { font-family: var(--head); font-size: 1.2rem; margin: 0; letter-spacing: -0.01em; }
    .slip-tear {
      position: relative; height: 0; border-top: 2px dashed var(--rule); margin: 0;
    }
    .slip-tear::before, .slip-tear::after {
      content: ""; position: absolute; top: -13px; width: 24px; height: 24px; border-radius: 50%;
      background: var(--mist);
    }
    .slip-tear::before { left: -12px; }
    .slip-tear::after { right: -12px; }
    .slip-body { padding: 1.75rem; min-height: 15rem; }

    .empty p { margin: 0; color: var(--slate); }
    .verdict {
      font-family: var(--head); font-weight: 700; font-size: clamp(1.9rem, 3.4vw, 2.5rem);
      line-height: 1.05; letter-spacing: -0.02em; margin: 0 0 1.25rem;
    }
    .state-approved .verdict { color: var(--pine); }
    .state-declined .verdict { color: var(--brick); }

    .meter-label { display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.4rem; }
    .meter { height: 14px; background: var(--mist); border-radius: 7px; overflow: hidden; }
    .meter > div { height: 100%; width: 0; border-radius: 7px; transition: width 0.7s cubic-bezier(.2,.7,.2,1); }
    .state-approved .meter > div { background: var(--pine); }
    .state-declined .meter > div { background: var(--brick); }
    .note { color: var(--slate); font-size: 0.875rem; margin: 1.25rem 0 0; }
    .problem { background: var(--brick-tint); color: var(--brick); border-radius: 8px; padding: 0.85rem 1rem; font-weight: 500; margin: 0; }

    .spinner { width: 1.1rem; height: 1.1rem; border-radius: 50%; border: 3px solid rgba(255,255,255,.4); border-top-color: #fff; display: inline-block; vertical-align: -3px; margin-right: 0.5rem; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    @media (max-width: 880px) {
      .layout { grid-template-columns: 1fr; gap: 1.5rem; }
      .slip-wrap { position: static; }
    }
    @media (max-width: 520px) {
      .page { padding-block: 1.5rem 3rem; }
      form { padding-inline: 1.1rem; }
      .grid { grid-template-columns: 1fr; }
      .actions { flex-direction: column-reverse; align-items: stretch; }
    }
    @media (prefers-reduced-motion: reduce) {
      * { transition: none !important; animation-duration: 0.01ms !important; }
    }
  </style>
</head>
<body>
  <div class="page">
    <header class="intro">
      <h1>See how your loan application is likely to go</h1>
      <p>Fill in the details below. The model compares them with past applications and estimates your chance of approval in a few seconds.</p>
    </header>

    {% macro seg(name, label, options, hint=None) %}
    <div class="field" data-field="{{ name }}">
      <span class="label" id="{{ name }}-label">{{ label }}</span>
      <div class="seg" role="radiogroup" aria-labelledby="{{ name }}-label">
        {% for value, text in options %}
          <input type="radio" id="{{ name }}-{{ loop.index }}" name="{{ name }}" value="{{ value }}" required>
          <label for="{{ name }}-{{ loop.index }}">{{ text }}</label>
        {% endfor %}
      </div>
      {% if hint %}<p class="hint">{{ hint }}</p>{% endif %}
      <p class="error" data-error-for="{{ name }}" role="alert"></p>
    </div>
    {% endmacro %}

    <div class="layout">
      <form id="loan-form" novalidate autocomplete="off">
        <fieldset>
          <legend>About the applicant</legend>
          <div class="grid">
            {{ seg("Gender", "Gender", [("Male", "Male"), ("Female", "Female")]) }}
            {{ seg("Married", "Married", [("Yes", "Yes"), ("No", "No")]) }}
            <div class="field" data-field="Dependents">
              <label for="Dependents">Dependents</label>
              <select id="Dependents" name="Dependents" required>
                <option value="" selected disabled>Select</option>
                <option value="0">None</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3+">3 or more</option>
              </select>
              <p class="error" data-error-for="Dependents" role="alert"></p>
            </div>
            {{ seg("Education", "Education", [("Graduate", "Graduate"), ("Not Graduate", "Not graduate")]) }}
            {{ seg("Self_Employed", "Self-employed", [("Yes", "Yes"), ("No", "No")]) }}
            <div class="field" data-field="Property_Area">
              <label for="Property_Area">Property area</label>
              <select id="Property_Area" name="Property_Area" required>
                <option value="" selected disabled>Select</option>
                <option value="Urban">Urban</option>
                <option value="Semiurban">Semi-urban</option>
                <option value="Rural">Rural</option>
              </select>
              <p class="error" data-error-for="Property_Area" role="alert"></p>
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>Income and credit</legend>
          <div class="grid">
            <div class="field" data-field="ApplicantIncome">
              <label for="ApplicantIncome">Your monthly income</label>
              <div class="prefix"><span aria-hidden="true">₹</span>
                <input type="number" id="ApplicantIncome" name="ApplicantIncome" min="0" step="1" inputmode="numeric" placeholder="5000" required>
              </div>
              <p class="error" data-error-for="ApplicantIncome" role="alert"></p>
            </div>
            <div class="field" data-field="CoapplicantIncome">
              <label for="CoapplicantIncome">Co-applicant’s monthly income</label>
              <div class="prefix"><span aria-hidden="true">₹</span>
                <input type="number" id="CoapplicantIncome" name="CoapplicantIncome" min="0" step="1" inputmode="numeric" placeholder="0" required>
              </div>
              <p class="hint">Enter 0 if there is no co-applicant.</p>
              <p class="error" data-error-for="CoapplicantIncome" role="alert"></p>
            </div>
            <div class="field wide">
              {{ seg("Credit_History", "Credit history", [("1", "Meets lender guidelines"), ("0", "Does not meet guidelines")], "Choose the first option if you have repaid past loans on time.") }}
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>The loan you want</legend>
          <div class="grid">
            <div class="field" data-field="LoanAmount">
              <label for="LoanAmount">Loan amount</label>
              <div class="prefix"><span aria-hidden="true">₹</span>
                <input type="number" id="LoanAmount" name="LoanAmount" min="1" step="1" inputmode="numeric" placeholder="150" required>
              </div>
              <p class="hint">In thousands. Enter 150 for ₹1,50,000.</p>
              <p class="error" data-error-for="LoanAmount" role="alert"></p>
            </div>
            <div class="field" data-field="Loan_Amount_Term">
              <label for="Loan_Amount_Term">Repayment period</label>
              <select id="Loan_Amount_Term" name="Loan_Amount_Term" required>
                <option value="12">1 year</option>
                <option value="36">3 years</option>
                <option value="60">5 years</option>
                <option value="84">7 years</option>
                <option value="120">10 years</option>
                <option value="180">15 years</option>
                <option value="240">20 years</option>
                <option value="300">25 years</option>
                <option value="360" selected>30 years</option>
                <option value="480">40 years</option>
              </select>
              <p class="error" data-error-for="Loan_Amount_Term" role="alert"></p>
            </div>
          </div>
        </fieldset>

        <div class="actions">
          <button type="submit" class="primary" id="submit-btn">Check eligibility</button>
          <button type="reset" class="ghost" id="reset-btn">Clear form</button>
        </div>
      </form>

      <aside class="slip-wrap" aria-label="Result">
        <div class="slip" id="slip">
          <div class="slip-top"><h2>Your result</h2></div>
          <div class="slip-tear" aria-hidden="true"></div>
          <div class="slip-body" id="slip-body" aria-live="polite">
            <div class="empty">
              <p>Fill in the form and select <strong>Check eligibility</strong>. Your estimate will appear here.</p>
            </div>
          </div>
        </div>
      </aside>
    </div>
  </div>

  <script>
    (function () {
      var form = document.getElementById("loan-form");
      var body = document.getElementById("slip-body");
      var slip = document.getElementById("slip");
      var btn = document.getElementById("submit-btn");

      var EMPTY = '<div class="empty"><p>Fill in the form and select <strong>Check eligibility</strong>. Your estimate will appear here.</p></div>';

      function clearErrors() {
        form.querySelectorAll(".field.invalid").forEach(function (f) { f.classList.remove("invalid"); });
        form.querySelectorAll("[data-error-for]").forEach(function (e) { e.textContent = ""; });
      }

      function showErrors(errors) {
        var first = null;
        Object.keys(errors).forEach(function (name) {
          var holder = form.querySelector('[data-error-for="' + name + '"]');
          var field = form.querySelector('[data-field="' + name + '"]');
          if (holder) holder.textContent = errors[name];
          if (field) { field.classList.add("invalid"); if (!first) first = field; }
        });
        if (first) {
          var target = first.querySelector("input, select");
          if (target) target.focus({ preventScroll: true });
          first.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }

      function localValidate(data) {
        var errors = {};
        ["Gender", "Married", "Education", "Self_Employed", "Credit_History"].forEach(function (k) {
          if (!data[k]) errors[k] = "Choose an option.";
        });
        ["Dependents", "Property_Area"].forEach(function (k) {
          if (!data[k]) errors[k] = "Select an option.";
        });
        [["ApplicantIncome", "Enter your monthly income."],
         ["CoapplicantIncome", "Enter 0 if there is no co-applicant."],
         ["LoanAmount", "Enter the loan amount in thousands."]].forEach(function (p) {
          if (data[p[0]] === "" || data[p[0]] == null) errors[p[0]] = p[1];
        });
        return errors;
      }

      function renderResult(result) {
        var cls = result.approved ? "state-approved" : "state-declined";
        var title = result.approved ? "Likely to be approved" : "Unlikely to be approved";
        var note = result.approved
          ? "Applications like yours were usually approved. A lender still makes the final decision."
          : "Applications like yours were usually declined. Credit history and the loan amount you ask for have the biggest effect, so changing them may help.";
        body.innerHTML =
          '<div class="' + cls + '">' +
            '<p class="verdict">' + title + '</p>' +
            '<div class="meter-label"><span>Approval chance</span><span>' + result.probability + '%</span></div>' +
            '<div class="meter" role="img" aria-label="Approval chance ' + result.probability + ' percent"><div></div></div>' +
            '<p class="note">' + note + ' This is an estimate from a statistical model, not a lending decision.</p>' +
          '</div>';
        var bar = body.querySelector(".meter > div");
        requestAnimationFrame(function () { requestAnimationFrame(function () { bar.style.width = result.probability + "%"; }); });
        if (window.matchMedia("(max-width: 880px)").matches) {
          slip.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }

      function renderProblem(message) {
        body.innerHTML = '<p class="problem">' + message + '</p>';
      }

      form.addEventListener("submit", function (e) {
        e.preventDefault();
        clearErrors();

        var data = {};
        new FormData(form).forEach(function (v, k) { data[k] = v; });

        var errors = localValidate(data);
        if (Object.keys(errors).length) {
          showErrors(errors);
          renderProblem("Some answers are missing. Complete the highlighted fields and try again.");
          return;
        }

        btn.disabled = true;
        btn.innerHTML = '<span class="spinner" aria-hidden="true"></span>Checking';

        fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data)
        })
          .then(function (res) { return res.json().then(function (json) { return { status: res.status, json: json }; }); })
          .then(function (r) {
            if (r.json.ok) { renderResult(r.json); }
            else if (r.json.errors) {
              showErrors(r.json.errors);
              renderProblem("Some answers need fixing. Check the highlighted fields and try again.");
            } else { renderProblem(r.json.message || "Something went wrong. Try again in a moment."); }
          })
          .catch(function () { renderProblem("Could not reach the server. Check your connection and try again."); })
          .then(function () { btn.disabled = false; btn.textContent = "Check eligibility"; });
      });

      form.addEventListener("reset", function () {
        clearErrors();
        body.innerHTML = EMPTY;
      });
    })();
  </script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    row, errors = parse_payload(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    try:
        frame = pd.DataFrame([row], columns=FEATURE_ORDER)
        approved_index = list(model.classes_).index(1)
        probability = float(model.predict_proba(frame)[0][approved_index])
        approved = bool(model.predict(frame)[0] == 1)
    except Exception:  # pragma: no cover - unexpected model failure
        log.exception("Prediction failed")
        return jsonify({"ok": False, "message": "The model could not score this application."}), 500

    return jsonify({
        "ok": True,
        "approved": approved,
        "probability": round(probability * 100, 1),
    })


@app.get("/health")
def health():
    """Health-check endpoint for AWS load balancers / Elastic Beanstalk."""
    return jsonify({"status": "ok"}), 200


@app.after_request
def security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
