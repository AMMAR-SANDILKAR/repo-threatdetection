# streamlit_threat_detection.py
# Threat detection frontend in Python using Streamlit + scikit-learn
# Single-file app: upload dataset or use synthetic data, train a classifier, show metrics, predict.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import joblib
import io

st.set_page_config(page_title="Threat Detection (ML)", layout="wide")

st.title("Threat Detection in Cybersecurity ")
st.markdown(
    """
    This lets you train a simple ML model (Random Forest) to classify network events as **benign** or **malicious**.

    - Upload a CSV with features and a target column named `label` (0 = benign, 1 = malicious), **or** use the built-in synthetic dataset.
    - Adjust hyperparameters, train, inspect metrics, and run predictions from the UI.
    """
)

# Sidebar controls
with st.sidebar:
    st.header("Dataset / Model")
    data_option = st.radio("Data source:", ("Use synthetic demo data", "Upload CSV"))
    if data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file (must contain a 'label' column)")
    else:
        uploaded_file = None

    st.markdown("---")
    st.subheader("Model hyperparameters")
    n_estimators = st.slider("Random Forest: n_estimators", 10, 500, 100)
    max_depth = st.slider("Random Forest: max_depth (None = unlimited)", 1, 50, 10)
    test_size = st.slider("Test set proportion", 10, 50, 20) / 100.0
    random_state = st.number_input("Random state (integer)", value=42, min_value=0)

    st.markdown("---")
    st.write("Model file")
    model_name = st.text_input("Save model filename", value="threat_detector.joblib")

# Load or create dataset
@st.cache_data
def create_synthetic(n_samples=2000, n_features=12, class_sep=1.0, flip_y=0.01, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.1),
        n_clusters_per_class=2,
        weights=[0.85, 0.15],
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )
    cols = [f"feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df['label'] = y
    return df


def load_user_csv(uploaded_file: io.BytesIO):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

    # Clean up column names (remove spaces, lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # Try to auto-detect the label column
    possible_labels = [c for c in df.columns if 'label' in c or 'class' in c or 'attack' in c]
    if not possible_labels:
        st.error("No label-like column found. Please ensure your dataset has a label or class column.")
        return None

    label_col = possible_labels[0]
    st.info(f"Detected label column: '{label_col}'")

    # Convert text labels to numeric (BENIGN=0, anything else=1)
    if df[label_col].dtype == object:
        df['label'] = df[label_col].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
    else:
        # If already numeric, rename it if necessary
        if label_col != 'label':
            df.rename(columns={label_col: 'label'}, inplace=True)

    # Ensure label is integer 0/1
    df['label'] = df['label'].astype(int)

    return df


# Main dataset selection
if uploaded_file is not None:
    df = load_user_csv(uploaded_file)
elif data_option == "Use synthetic demo data":
    df = create_synthetic()
else:
    df = None

if df is None:
    st.warning("No dataset available. Upload a CSV or select synthetic demo data from the sidebar.")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(df.head(10))

# Feature / target separation
all_columns = df.columns.tolist()
if 'label' not in all_columns:
    st.error("Required column 'label' missing.")
    st.stop()
feature_cols = [c for c in all_columns if c != 'label']
st.write(f"Detected features: {len(feature_cols)} columns")

# Option to pick features
with st.expander("Select features to use (autoselect all by default)", expanded=False):
    selected_features = st.multiselect("Features", options=feature_cols, default=feature_cols)

if len(selected_features) == 0:
    st.error("Select at least one feature.")
    st.stop()

X = df[selected_features].copy()
y = df['label'].copy()

# Train/test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state), stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train model on button press
if st.button("Train Random Forest"):
    with st.spinner("Training model..."):
        clf = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=None if max_depth <= 0 else int(max_depth), random_state=int(random_state))
        clf.fit(X_train_s, y_train)

        # predictions and metrics
        preds = clf.predict(X_test_s)
        probs = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, 'predict_proba') else None

        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        st.success("Training complete")

        # Show basic metrics
        st.subheader("Performance")
        st.write("**Classification report (test set)**")
        st.dataframe(pd.DataFrame(report).transpose())

        st.write("**Confusion matrix**")
        cm_fig, ax = plt.subplots()
        ax.imshow(cm, interpolation='nearest')
        ax.set_title('Confusion matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha='center', va='center', color='white' if val > cm.max()/2 else 'black')
        st.pyplot(cm_fig)

        if probs is not None:
            try:
                auc = roc_auc_score(y_test, probs)
                st.write(f"ROC AUC: **{auc:.4f}**")
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_fig, ax2 = plt.subplots()
                ax2.plot(fpr, tpr)
                ax2.plot([0,1], [0,1], linestyle='--')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.set_title('ROC Curve')
                st.pyplot(roc_fig)
            except Exception as e:
                st.write("Could not compute ROC AUC: ", e)

        # Feature importances
        if hasattr(clf, 'feature_importances_'):
            fi = pd.Series(clf.feature_importances_, index=selected_features).sort_values(ascending=False)
            st.write("**Feature importances**")
            st.dataframe(fi.head(20))

        # Save model + scaler
        b = io.BytesIO()
        joblib.dump({'model': clf, 'scaler': scaler, 'features': selected_features}, model_name)
        st.write(f"Model saved to `{model_name}` in the working directory.")

        # Provide single-sample prediction interface
        with st.expander("Make a prediction on custom input"):
            st.write("Enter feature values or use random sample from test set")
            if st.button("Use random test sample"):
                sample_idx = np.random.randint(0, X_test.shape[0])
                sample = X_test.iloc[sample_idx]
            else:
                sample = None

            input_vals = []
            cols = selected_features
            cols_inputs = {}
            for c in cols:
                val = st.text_input(f"{c}", value=str(sample[c]) if sample is not None else "0.0")
                try:
                    cols_inputs[c] = float(val)
                except:
                    cols_inputs[c] = 0.0

            if st.button("Predict sample"):
                sample_df = pd.DataFrame([cols_inputs])
                sample_s = scaler.transform(sample_df)
                pred = clf.predict(sample_s)[0]
                prob = clf.predict_proba(sample_s)[0][1] if hasattr(clf, 'predict_proba') else None
                label_map = {0: 'Benign', 1: 'Malicious'}
                st.write(f"Prediction: **{label_map.get(pred, pred)}**")
                if prob is not None:
                    st.write(f"Malicious probability: **{prob:.3f}**")

# If not trained yet, show quick EDA
else:
    st.info("Model not trained yet — press 'Train Random Forest' in the sidebar to begin.")
    st.subheader("Quick dataset stats")
    st.write(df[selected_features].describe())
    st.write("Label distribution:")
    st.write(df['label'].value_counts(normalize=True))

    with st.expander("Show correlation matrix"):
        corr = df[selected_features].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        ax.set_xticks(range(len(selected_features)))
        ax.set_yticks(range(len(selected_features)))
        ax.set_xticklabels(selected_features, rotation=90)
        ax.set_yticklabels(selected_features)
        st.pyplot(fig)

# Footer / notes
st.markdown("---")
st.caption(" For production-ready threat detection: use vetted datasets (e.g., CICIDS, UNSW-NB15), handle feature engineering for packet/payload fields, follow security and privacy best practices, and validate on realistic traffic.")
