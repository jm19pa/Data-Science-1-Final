import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from DatasetInfo import X_train_scaled, X_test_scaled, y_train, y_test, label_encoder

# # Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000,multi_class='multinomial')
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

print("=" * 80)
print("Logistic Regression Results")
print("=" * 80)
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log_reg, target_names=label_encoder.classes_, digits=4))

#Logistic Regression Confusion Matrix
pred_genres_xgb = label_encoder.inverse_transform(y_pred_log_reg)
true_genres = label_encoder.inverse_transform(y_test)
results_df = pd.DataFrame({
    "True Genre": true_genres,
    "Predicted Genre": pred_genres_xgb
})
print(results_df.head(10))

#classificatipon report
report = classification_report(y_test, y_pred_log_reg, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
genre_report = report_df.iloc[:-3, :]  
genre_report['f1-score'].plot(kind='bar', figsize=(15, 5))
plt.title('F1-Score by Genre - Logistic Regression')
plt.xlabel('Genre')
plt.ylabel('F1-Score')
plt.xticks(rotation=90, ha='right', fontsize=5)
plt.show()

