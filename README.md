# Telco Churn — API + Streamlit UI

Two small services, tested end-to-end in this environment:

- **`api/`** — FastAPI service that loads the trained pipeline once and
  exposes `POST /predict` and `GET /features`.
- **`streamlit_app/`** — Streamlit UI that calls the API over HTTP. It never
  touches the `.pkl` files directly, so it can be deployed completely
  separately from the model.

## ⚠️ Critical finding — you must read this before deploying

The delivered `pipeline2.pkl` contains a custom transformer
(`SeniorCitizenTransformer`) that was saved with `dill` from a **Google Colab
notebook running Python 3.10.5**. `dill` embeds the transformer's raw
bytecode, and Python bytecode is **not compatible across major/minor CPython
versions**.

I verified this directly:
- Loading + predicting with this pickle under **Python 3.12** loads without
  any error, but produces a corrupted result (`AttributeError: 'list' object
  has no attribute 'isinstance'` raised deep inside sklearn's internals) —
  the kind of failure that could easily be shipped to the client unnoticed
  if only "does it load" was checked.
- Loading + predicting under **Python 3.10.20** with the exact package
  versions in `api/requirements.txt` gives correct, sane predictions (I
  cross-checked several rows from the training CSV against direct pipeline
  calls — probabilities matched exactly).

**Action required:** deploy the API on Python 3.10.x only. I've pinned this
three ways so it's hard to get wrong by accident:
- `api/Dockerfile` uses `python:3.10-slim`
- `api/runtime.txt` (`python-3.10.13`) for platforms that read it (Render,
  Heroku-style buildpacks)
- `api/requirements.txt` pins the exact library versions I tested against
  (`scikit-learn==1.4.1.post1`, `dill==0.3.8`, `joblib==1.3.2`,
  `pandas==2.2.3`)

If you (or the freelancer) ever retrain the model, the safest long-term fix
is to re-save the pipeline with `dill` under whatever Python version you'll
deploy on — or better, replace the custom transformer with a
`sklearn.compose.ColumnTransformer` + `FunctionTransformer` combo (no custom
class), which is more portable. Not required now — just flagging it since
you have limited hours and this is the one thing that will silently break
if someone "helpfully" upgrades the Python version later.

Two smaller things I also found and cleaned up:
- The original `requirements.txt` had a duplicate `joblib` line and a typo
  (`~treamlit` instead of `streamlit`) — fixed in the new requirements files.
- `my_feature_dict.pkl`'s `CATEGORICAL` block lists `SENIORCITIZEN` twice
  (a quirk from how it was exported) — the API's `/features` endpoint
  de-duplicates this automatically so the UI form doesn't show a repeated
  field.
- The `RandomForestClassifier` was trained without a `random_state`, so a
  retrain will give slightly different results each time — not a bug, just
  worth mentioning if the client asks about reproducibility.

## Running locally

**API** (needs Python 3.10.x):
```bash
cd api
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Check it: `curl http://localhost:8000/health` and `http://localhost:8000/docs`
for interactive Swagger docs.

**Streamlit UI** (any Python 3.9+ is fine — it has no model dependencies):
```bash
cd streamlit_app
pip install -r requirements.txt
API_URL=http://localhost:8000 streamlit run app.py
```

## Deploying (fits comfortably in a few hours)

1. **API → Render.com (free tier) using the Dockerfile**
   - New Web Service → connect your repo → root directory `api/`
   - Render detects the `Dockerfile` automatically, so the Python 3.10 pin
     is respected regardless of Render's own defaults.
   - Note the public URL, e.g. `https://your-api.onrender.com`.
   - (Railway or Fly.io work the same way if you prefer those.)

2. **UI → Streamlit Community Cloud (free)**
   - New app → point at `streamlit_app/app.py` in your repo.
   - In the app's **Settings → Secrets**, add:
     ```
     API_URL = "https://your-api.onrender.com"
     ```
   - Deploy. The UI will now call your live API for every prediction —
     no `.pkl` files anywhere near the Streamlit app.

3. Free-tier note: Render's free web services sleep after inactivity and
   take ~30-50s to wake on the first request — mention this to the client
   if it matters, or upgrade the plan if not acceptable.

## API reference

- `GET /health` → `{"status": "ok"}`
- `GET /features` → categorical field names + their allowed options, and
  the numerical field names — this is what the Streamlit UI uses to build
  its form dynamically.
- `POST /predict` → body is a JSON object with all 19 raw customer fields
  (see `api/main.py`'s `CustomerInput` model, or `/docs` for the schema).
  Returns:
  ```json
  {
    "prediction": "No",
    "churn_probability": 0.0435,
    "probabilities": {"No": 0.9565, "Yes": 0.0435}
  }
  ```
