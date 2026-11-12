# AI-Based Carbon Emission Prediction and Reduction – Streamlit deployment

| File | Role |
|---|---|
| `carbon_emission_dataset_with_Industry.csv` | dataset (18,250 rows) |
| `preprocessing.py` | load, drop ID + leakage columns, 80/20 split (seed 42), validate queries |
| `encoding.py` | one-hot encoding (13 numeric + 17 dummy = 30 features) |
| `scaling.py` | StandardScaler fitted on the training set |
| `model.py` | LASSO / RF / hybrid selection, 15 boosting models, CV selection, what-if, saves pickle |
| `carbon_model.pickle` | trained pipeline (encoder + scaler + selected features + best model) |
| `app.py` | Streamlit interface: every query goes through the pickled model |

## Run locally
    pip install -r requirements.txt
    python model.py          # optional: retrain and rebuild carbon_model.pickle
    streamlit run app.py

## Deploy (free)
Streamlit needs a Python server, so it cannot run *on* GitHub Pages itself.
1. Push this folder to a GitHub repo.
2. https://share.streamlit.io -> New app -> pick the repo, main file `app.py`.
3. Put the resulting URL into `docs/index.html` (`APP_URL`), and enable GitHub Pages on the `/docs` folder.
   The Pages site then embeds / links to the live app.
