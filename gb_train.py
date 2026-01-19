import pandas as pd
import numpy as np
import pickle

# sklearn preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Regression models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Cross-validation and tuning
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV


# =====================
# Load dataset
# =====================
df = pd.read_csv("medical_insurance.csv")

print(df)

print(f"Original dataset shape: {df.shape}")
print(f"Duplicates found: {df.duplicated().sum()}")

# Remove duplicates
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")


# Target and features
X = df.drop('charges', axis=1)
y = df['charges']

# =====================
# Column split
# =====================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# =====================
# Preprocessing
# =====================
num_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

# Categorical transformer - impute + encode
cat_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

# =====================
# Random Forest Model
# =====================
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

# =====================
# Full Pipeline
# =====================
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', gb_model)
])

# =====================
# Train-test split
# ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


gb_pipeline.fit(X_train, y_train)

# =====================
# Evaluation
# =====================
y_pred = gb_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# =====================
# Save model (IMPORTANT)
# =====================

with open("insurance_gb_pipeline.pkl", "wb") as f:
    pickle.dump(gb_pipeline, f)

print("Gradient Boosting pipeline saved as insurance_gb_pipeline.pkl")