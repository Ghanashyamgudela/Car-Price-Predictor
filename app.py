import streamlit as st
import pandas as pd
import pickle
import numpy as np
import scipy.sparse as sp
import sklearn.compose._column_transformer as _ct

# Compatibility shim: some pickled models reference
# `_RemainderColsList` in `sklearn.compose._column_transformer`.
# If that symbol is missing (different scikit-learn version),
# define a lightweight placeholder so `pickle.load` succeeds.
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Load model
model = pickle.load(open('RandomForestRegressor.pkl', 'rb'))


def predict_with_padding(model, input_df):
    """Predict robustly by using the pipeline preprocessing when present,
    converting nested/sparse outputs to a numeric 2D array, and padding or
    truncating to match the final estimator's expected feature count.
    Returns a single prediction value or None on diagnostic failure.
    """
    # Fast path: try model.predict (handles many pipeline shapes)
    try:
        return model.predict(input_df)[0]
    except Exception:
        pass

    # If pipeline-like, separate preprocessing and final estimator
    if hasattr(model, "steps") and len(model.steps) >= 1:
        preproc = model[:-1]
        final_est = model.steps[-1][1]
        try:
            Xt = preproc.transform(input_df)
        except Exception as e:
            msg = str(e)
            if "columns are missing" in msg:
                try:
                    import ast
                    missing = ast.literal_eval(msg.split("columns are missing:")[1].strip())
                except Exception:
                    missing = None
                if isinstance(missing, (set, list, tuple)):
                    for col in missing:
                        input_df[col] = 0
                    Xt = preproc.transform(input_df)
                else:
                    raise
            else:
                raise

        # Convert Xt into numeric 2D ndarray
        try:
            # sparse -> dense
            if sp.issparse(Xt):
                Xt = Xt.toarray()

            # pandas DataFrame -> ndarray
            try:
                import pandas as _pd
                if isinstance(Xt, _pd.DataFrame):
                    Xt = Xt.values
            except Exception:
                pass

            # Common case observed: array([[<sparse matrix>]], dtype=object)
            if isinstance(Xt, np.ndarray) and Xt.dtype == object and Xt.size == 1:
                elem = Xt.flat[0]
                if sp.issparse(elem):
                    Xt = np.asarray(elem.toarray(), dtype=float)
                else:
                    Xt = np.asarray(elem, dtype=float)
                if Xt.ndim == 1:
                    Xt = Xt.reshape(1, -1)
            else:
                # If already numeric ndarray
                if isinstance(Xt, np.ndarray) and Xt.dtype != object:
                    Xt = np.asarray(Xt, dtype=float)
                    if Xt.ndim == 1:
                        Xt = Xt.reshape(1, -1)
                else:
                    # Try to handle list/tuple of parts
                    if isinstance(Xt, (list, tuple)):
                        parts = []
                        for part in Xt:
                            if sp.issparse(part):
                                parts.append(np.asarray(part.toarray(), dtype=float))
                            else:
                                arr = np.asarray(part)
                                if arr.dtype == object:
                                    # try to extract nested element
                                    if arr.size == 1:
                                        arr = np.asarray(arr.flat[0], dtype=float)
                                    else:
                                        arr = arr.astype(float)
                                parts.append(arr.reshape(1, -1) if arr.ndim == 1 else arr)
                        Xt = np.hstack(parts)
                        if Xt.ndim == 1:
                            Xt = Xt.reshape(1, -1)
                    else:
                        # give up and raise to produce diagnostics
                        raise ValueError("Unhandled transformer output type")
        except Exception:
            # Provide diagnostics in Streamlit and return None
            try:
                import streamlit as _st
                diag = {"type": str(type(Xt))}
                try:
                    diag["repr"] = repr(Xt)[:2000]
                except Exception:
                    diag["repr"] = "<unrepresentable>"
                try:
                    diag["shape"] = getattr(Xt, 'shape', None)
                    diag["dtype"] = getattr(Xt, 'dtype', None)
                except Exception:
                    pass
                _st.error("Could not convert transformer output to numeric 2D array. See diagnostics below.")
                _st.write(diag)
                try:
                    _st.info("Attempting fallback: pipeline.predict(input_df)")
                    _st.write(model.predict(input_df))
                except Exception as e_fallback:
                    _st.write({"fallback_error": repr(e_fallback)})
            except Exception:
                pass
            return None

        # Pad or truncate to expected feature count
        expected = getattr(final_est, "n_features_in_", None)
        if expected is not None:
            if Xt.shape[1] < expected:
                pad = np.zeros((Xt.shape[0], expected - Xt.shape[1]))
                Xt = np.hstack([Xt, pad])
            elif Xt.shape[1] > expected:
                Xt = Xt[:, :expected]

        return final_est.predict(Xt)[0]

    # Non-pipeline estimators
    expected = getattr(model, "n_features_in_", None)
    input_cat = pd.get_dummies(input_df[['name', 'company', 'fuel_type']])
    input_num = input_df[['year', 'kms_driven']].astype(float)
    input_proc = pd.concat([input_num, input_cat], axis=1)
    arr = input_proc.to_numpy()
    if expected is not None:
        if arr.shape[1] < expected:
            pad = np.zeros((arr.shape[0], expected - arr.shape[1]))
            arr = np.hstack([arr, pad])
        elif arr.shape[1] > expected:
            arr = arr[:, :expected]
    return model.predict(arr)[0]

# Load data
car = pd.read_csv('Cleaned_Car_data.csv')

# Build reference feature columns (one-hot columns) from training data
# so user input can be encoded to the same feature vector shape expected
# by the saved model.
_train_dummies = pd.get_dummies(car[['name', 'company', 'fuel_type']])
FEATURE_COLUMNS = ['year', 'kms_driven'] + list(_train_dummies.columns)

st.title("🚗 Car Price Prediction Dashboard")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", sorted(car['company'].unique()))
    year = st.selectbox("Year", sorted(car['year'].unique(), reverse=True))
    kms = st.number_input("Kilometers Driven", min_value=0)

with col2:
    name = st.selectbox("Model", car['name'].unique())
    fuel = st.selectbox("Fuel Type", car['fuel_type'].unique())

if st.button("Predict Price 💰"):
    input_df = pd.DataFrame(
        [[name, company, year, kms, fuel]],
        columns=['name','company','year','kms_driven','fuel_type']
    )

    prediction = predict_with_padding(model, input_df)

    if prediction is None:
        st.error("Prediction failed — diagnostics shown above. Paste the diagnostics here and I'll help fix it.")
    else:
        st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")

# ---- Stylish Sidebar & Data Disclaimer ----
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="color: #007BFF; margin-bottom: 0;">🚙 Car Analysis</h2>
    <p style="color: #999999; font-size: 14px; margin-top: 5px;">Advanced ML Predictor</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📊 About the App")
st.sidebar.info(
    "This application uses a Machine Learning model to predict the estimated price of a car "
    "based on its specifications."
)

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚠️ Data Disclaimer")
st.sidebar.warning(
    "The predictions provided by this dashboard are based on historical data. "
    "Actual market prices may vary due to external factors. "
    "Please use this tool as a reference rather than a definitive guide."
)

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='text-align: center; color: #555;'><small>Built for premium insights</small></div>", unsafe_allow_html=True)

# ---- Custom CSS for Background and Footer ----
st.markdown("""
    <style>
    /* Gradient Background Effect */
    .stApp {
        background-color: #000000;
        background-image: 
            radial-gradient(circle at 20% 40%, rgba(0, 123, 255, 0.15) 0%, transparent 40%), 
            radial-gradient(circle at 80% 60%, rgba(0, 123, 255, 0.1) 0%, transparent 40%);
        background-attachment: fixed;
    }

    /* Custom Footer CSS */
    .custom-footer {
        background-color: transparent;
        color: #cccccc;
        padding: 40px 20px 20px 20px;
        margin-top: 50px;
        border-top: 1px solid #222222;
        font-family: sans-serif;
    }

    .footer-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        max-width: 1000px;
        margin: 0 auto;
    }

    .footer-col {
        flex: 1;
        min-width: 200px;
        margin-bottom: 20px;
    }

    .footer-col h4 {
        color: #ffffff;
        font-size: 16px;
        margin-bottom: 20px;
        font-weight: 600;
    }

    .footer-col ul {
        list-style-type: none;
        padding: 0;
        margin: 0;
    }

    .footer-col ul li {
        margin-bottom: 12px;
    }

    .footer-col ul li a {
        color: #999999;
        text-decoration: none;
        font-size: 14px;
        transition: 0.3s;
    }

    .footer-col ul li a:hover {
        color: #007BFF;
    }

    .footer-bottom {
        text-align: center;
        padding-top: 20px;
        border-top: 1px solid #333333;
        color: #777777;
        font-size: 13px;
        margin-top: 20px;
    }
    </style>

    <div class="custom-footer">
        <div class="footer-container">
            <div class="footer-col">
                <h4>Get to Know Us</h4>
                <ul>
                    <li><a href="#" target="_blank">About Me</a></li>
                    <li><a href="#" target="_blank">Portfolio Overview</a></li>
                    <li><a href="https://colab.research.google.com/drive/1U6CXuLdps80eACQWKal02eMEjOt0HXqu?usp=sharing" target="_blank">Data Insights</a></li>
                </ul>
            </div>
            <div class="footer-col">
                <h4>Connect with Me</h4>
                <ul>
                    <li><a href="www.linkedin.com/in/ghana-shyam-gudela-151a23295" target="_blank">LinkedIn</a></li>
                    <li><a href="https://github.com/Ghanashyamgudela" target="_blank">GitHub</a></li>
                    <li><a href="#" target="_blank">Twitter (X)</a></li>
                </ul>
            </div>
            <div class="footer-col">
                <h4>Let Us Help You</h4>
                <ul>
                    <li><a href="mailto:ghana19183@gmail.com">Contact Support</a></li>
                    <li><span style="color: #999999; font-size: 14px; cursor: default;">Data Disclaimer (See Sidebar)</span></li>
                </ul>
            </div>
        </div>
        <div class="footer-bottom">
            © 2025 Ghana Shyam Gudela. All Rights Reserved. Data dashboard designed for premium insights.
        </div>
    </div>
""", unsafe_allow_html=True)
