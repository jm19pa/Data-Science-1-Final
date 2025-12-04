import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

from DatasetInfo import X_train_scaled, X_test_scaled, y_train, y_test, label_encoder, feature_cols

#XGBoost Classifier Model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

print("XGBoost Classifier Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classifier Classification Report:\n", classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))
# print("XGBoost Classifier Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

pred_genres_xgb = label_encoder.inverse_transform(y_pred_xgb)
true_genres = label_encoder.inverse_transform(y_test)
results_df = pd.DataFrame({
    "True Genre": true_genres,
    "Predicted Genre": pred_genres_xgb
})
print(results_df.head(10))

#classificatipon report
report = classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
genre_report = report_df.iloc[:-3, :]  
genre_report['f1-score'].plot(kind='bar', figsize=(15, 5))
plt.title('F1-Score by Genre - XGBoost Classifier')
plt.xlabel('Genre')
plt.ylabel('F1-Score')
plt.xticks(rotation=90, ha='right', fontsize=5)
plt.show()


