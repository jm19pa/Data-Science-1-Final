import FinalMain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = FinalMain.df.dropna()
target_col = FinalMain.target_col
feature_cols = FinalMain.feature_cols

genre_counts = df[target_col].value_counts()
print("Genre distribution:")
print(genre_counts, "\n")

#bar chart is here but they all have 1000 tracks
# plt.figure(figsize=(8,4))
# genre_counts.plot(kind='bar')
# plt.title("Number of Tracks per Genre")
# plt.xlabel("Genre")
# plt.ylabel("Number of Tracks")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

#histograms for features
df[feature_cols].hist(bins=30, figsize=(15,10))
plt.suptitle("Feature Distributions", y= 1.02, fontsize=16)
plt.tight_layout()
plt.show()

#correlation heatmap
corr = df[feature_cols].corr()
plt.figure(figsize=(10,8))
im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(ticks=np.arange(len(feature_cols)), labels=feature_cols, rotation=45, ha="right")
plt.yticks(ticks=np.arange(len(feature_cols)), labels=feature_cols)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()