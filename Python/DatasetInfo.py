import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("dataset.csv").dropna()


feature_cols = [
    "danceability", "energy", "loudness", "tempo",
    "acousticness", "valence", "speechiness",
    "instrumentalness", "liveness", "duration_ms"
]

target_col = "track_genre"
# Expose the target column variable; printing and plotting are done only when
# this file is executed as a script (see bottom `if __name__ == '__main__'`).

X = df[feature_cols]
y = df[target_col]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# print("Classes:", label_encoder.classes_)
# training data split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
# scaling data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

genre_counts = df[target_col].value_counts()
# `genre_counts` is available if another module wants it; avoid printing on import




if __name__ == '__main__':
    # Print some basic dataset info and show exploratory plots when run directly
    print(target_col, "\n")

    print("Genre distribution:")
    print(genre_counts, "\n")

    # histograms for features
    df[feature_cols].hist(bins=30, figsize=(15,10))
    plt.suptitle("Feature Distributions", y= 1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

    # correlation heatmap (showed after histograms)
    corr = df[feature_cols].corr()
    plt.figure(figsize=(10,8))
    im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(ticks=np.arange(len(feature_cols)), labels=feature_cols, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(feature_cols)), labels=feature_cols)
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()

    #bar chart is here but they all have 1000 tracks making it unnecessary
    # plt.figure(figsize=(8,4))
    # genre_counts.plot(kind='bar')
    # plt.title("Number of Tracks per Genre")
    # plt.xlabel("Genre")
    # plt.ylabel("Number of Tracks")
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.show()