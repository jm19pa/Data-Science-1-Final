import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

df = pd.read_csv("dataset.csv")

# print(df.head())
# print(df.info())

# Drop rows with missing values
df = df.dropna()


feature_cols = [
    "danceability", "energy", "loudness", "tempo",
    "acousticness", "valence", "speechiness",
    "instrumentalness", "liveness", "duration_ms"
]

target_col = "track_genre"

# X = df[feature_cols]
# y = df[target_col]

# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# # print("Classes:", label_encoder.classes_)
# # training data split
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
# # scaling data 
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Logistic Regression Model
# log_reg = LogisticRegression(max_iter=1000,multi_class='multinomial')
# log_reg.fit(X_train_scaled, y_train)
# y_pred_log_reg = log_reg.predict(X_test_scaled)

# print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
# print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))


# # XGBoost Classifier Model
# xgb_model = XGBClassifier(
#     n_estimators=200,
#     max_depth=6,
#     learning_rate=0.1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective='multi:softmax',
#     num_class=len(label_encoder.classes_),
#     random_state=42
# )

# xgb_model.fit(X_train_scaled, y_train)
# y_pred_xgb = xgb_model.predict(X_test_scaled)

# print("XGBoost Classifier Accuracy:", accuracy_score(y_test, y_pred_xgb))
# print("XGBoost Classifier Classification Report:\n", classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))
# # print("XGBoost Classifier Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# pred_genres_xgb = label_encoder.inverse_transform(y_pred_xgb)
# true_genres = label_encoder.inverse_transform(y_test)
# results_df = pd.DataFrame({
#     "True Genre": true_genres,
#     "Predicted Genre": pred_genres_xgb
# })
# print(results_df.head(10))
