import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib

# Load labeled dataset
df = pd.read_csv(r'D:\SEM 6\gait\data\gait_angles_featurelabels.csv')

# Feature engineering
angle_cols = ['hip_angle_r', 'knee_angle_r', 'foot_strike_angle_r',
              'hip_angle_l', 'knee_angle_l', 'foot_strike_angle_l', 'torso_angle']

# Add symmetry features
df['hip_symmetry'] = abs(df['hip_angle_r'] - df['hip_angle_l'])
df['knee_symmetry'] = abs(df['knee_angle_r'] - df['knee_angle_l'])
df['foot_symmetry'] = abs(df['foot_strike_angle_r'] - df['foot_strike_angle_l'])

# Create overall label (1 if all angles correct, else 0)
status_cols = [f"{col}_status" for col in angle_cols]
df['overall_label'] = df[status_cols].min(axis=1)  # All must be 1 for overall=1

# Remove rows with NaN
df = df.dropna(subset=angle_cols)

# Features for training
feature_cols = angle_cols + ['hip_symmetry', 'knee_symmetry', 'foot_symmetry']
X = df[feature_cols].values
y = df['overall_label'].values

print(f"Dataset size: {len(df)}")
print(f"Correct frames: {sum(y==1)}, Incorrect frames: {sum(y==0)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost (fast + accurate)
print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("\n=== MODEL EVALUATION ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Incorrect', 'Correct']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature importance
importances = model.feature_importances_
print("\n=== FEATURE IMPORTANCE ===")
for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp:.4f}")

# Save model and scaler
joblib.dump(model, r'D:\SEM 6\gait\treadmill_posture_model.pkl')
joblib.dump(scaler, r'D:\SEM 6\gait\treadmill_scaler.pkl')
print("\n✓ Model and scaler saved!")
