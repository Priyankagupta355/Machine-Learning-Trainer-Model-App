import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import pickle

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# 1. Welcome message
st.title("🧠 Machine Learning Model Trainer App")
st.write("Welcome! Upload data or use example datasets and train ML models with automatic preprocessing.")

# 2. Data source selection
use_upload = st.sidebar.radio("Do you want to upload your own dataset?", ("Yes", "No"))

if use_upload == "Yes":
    uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV, XLSX, TSV)", type=['csv', 'xlsx', 'tsv'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.tsv'):
            sep = ',' if uploaded_file.name.endswith('.csv') else '\t'
            df = pd.read_csv(uploaded_file, sep=sep)
        else:
            df = pd.read_excel(uploaded_file)
    else:
        st.warning("Please upload a dataset.")
        st.stop()
else:
    dataset_name = st.sidebar.selectbox("Select Example Dataset", ("titanic", "iris", "tips"))
    df = sns.load_dataset(dataset_name)

# 3. Dataset Overview
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.write(f"Shape: {df.shape}")
st.write(f"Columns: {df.columns.tolist()}")
st.write("Description:")
st.write(df.describe())

buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

# 4. Feature and Target Selection
all_columns = df.columns.tolist()
feature_cols = st.multiselect("Select Feature Columns", all_columns)
target_col = st.selectbox("Select Target Column", all_columns)

if not feature_cols or not target_col:
    st.warning("Please select feature columns and target column.")
    st.stop()

X = df[feature_cols]
y = df[target_col]

# 5. Problem Type Detection
problem_type = "regression" if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20 else "classification"
st.info(f"This looks like a {problem_type.upper()} problem.")

# 6. Preprocessing
st.subheader("Preprocessing Data")

# Separate numeric and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Show original missing data summary
st.write("🔍 Missing values before imputation:")
st.write(X.isnull().sum())

# Numeric imputer
if len(num_cols) > 0:
    num_imputer = IterativeImputer()
    X.loc[:, num_cols] = num_imputer.fit_transform(X[num_cols])
else:
    st.info("No numeric columns to impute.")

# Categorical imputer
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X.loc[:, cat_cols] = cat_imputer.fit_transform(X[cat_cols])
else:
    st.info("No categorical columns to impute.")

# Show missing data after imputation
st.write("✅ Missing values after imputation:")
st.write(X.isnull().sum())

# Encode categorical features
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Scale features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
# Separate numeric and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Show original missing data summary
st.write("🔍 Missing values before imputation:")
st.write(X.isnull().sum())

# Numeric imputer
num_imputer = IterativeImputer()
X.loc[:, num_cols] = num_imputer.fit_transform(X[num_cols])
if len(num_cols) > 0:
    X.loc[:,num_cols] = num_imputer.fit_transform(X[num_cols])

# Categorical imputer
cat_imputer = SimpleImputer(strategy='most_frequent')
X.loc[:, cat_cols] = cat_imputer.fit_transform(X[cat_cols])
if len(cat_cols) > 0:
    X.loc[:,cat_cols] = cat_imputer.fit_transform(X[cat_cols])
# Show missing data after imputation
st.write("✅ Missing values after imputation:")
st.write(X.isnull().sum())

# Encode categorical features
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Scale features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Encode target if classification
target_encoder = None
if problem_type == "classification" and y.dtype == 'object':
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Selection
st.subheader("Select Model")
if problem_type == "regression":
    model_options = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Support Vector Regressor": SVR()
    }
else:
    model_options = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "Support Vector Classifier": SVC(probability=True)
    }

selected_model_name = st.selectbox("Choose Model", list(model_options.keys()))
model = model_options[selected_model_name]

# 8. Train Model
st.subheader("Training Model...")
model.fit(X_train, y_train)
st.success("Model trained successfully!")

# 9. Model Evaluation
st.subheader("Model Evaluation")
y_pred = model.predict(X_test)

if problem_type == "regression":
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"✅ MSE: {mse:.4f}")
    st.write(f"✅ RMSE: {rmse:.4f}")
    st.write(f"✅ MAE: {mae:.4f}")
    st.write(f"✅ R2 Score: {r2:.4f}")

else:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.write(f"✅ Accuracy: {acc:.4f}")
    st.write(f"✅ Precision: {prec:.4f}")
    st.write(f"✅ Recall: {rec:.4f}")
    st.write(f"✅ F1 Score: {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

# 10. Model Download
if st.checkbox("Download trained model as .pkl"):
    buffer = io.BytesIO()
    pickle.dump({
        "model": model,
        "scaler": scaler,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "encoders": encoders,
        "target_encoder": target_encoder,
        "feature_cols": feature_cols
    }, buffer)
    buffer.seek(0)
    st.download_button(
        label="Download Model (.pkl)",
        data=buffer,
        file_name="trained_model.pkl",
        mime="application/octet-stream"
    )

# 11. New Prediction
if st.checkbox("Make a new prediction"):
    st.write("Provide feature values:")
    input_data = {}
    for col in feature_cols:
        value = st.number_input(f"{col}", value=0.0)
        input_data[col] = value

    input_df = pd.DataFrame([input_data])

    # Apply preprocessing
    input_df.loc[:, num_cols] = num_imputer.transform(input_df[num_cols])
    input_df.loc[:, cat_cols] = cat_imputer.transform(input_df[cat_cols])
    for col in cat_cols:
        input_df[col] = encoders[col].transform(input_df[col].astype(str))

    input_df = pd.DataFrame(scaler.transform(input_df), columns=feature_cols)

    prediction = model.predict(input_df)
    if problem_type == "classification" and target_encoder:
        prediction = target_encoder.inverse_transform(prediction.astype(int))

    st.success(f"🔮 Predicted Output: {prediction[0]}")