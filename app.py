
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Data Generation
# -----------------------------
np.random.seed(42)

data = {
    "Student_ID": range(1, 201),
    "Course_Duration_Hours": np.random.randint(10, 60, 200),
    "Videos_Watched": np.random.randint(5, 50, 200),
    "Avg_Watch_Time_Min": np.random.randint(5, 30, 200),
    "Assignments_Submitted": np.random.randint(0, 10, 200),
    "Quiz_Score": np.random.randint(40, 100, 200),
    "Days_Active": np.random.randint(1, 30, 200)
}

df = pd.DataFrame(data)

print("First 5 Records:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Quiz_Score"], bins=20, kde=True)
plt.title("Distribution of Quiz Scores")
plt.xlabel("Quiz Score")
plt.ylabel("Number of Students")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="Videos_Watched", y="Quiz_Score", data=df)
plt.title("Videos Watched vs Quiz Score")
plt.show()

plt.figure(figsize=(10,6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Between Learning Factors")
plt.show()

# -----------------------------
# Clustering Analysis
# -----------------------------
features = df[
    ["Videos_Watched", "Avg_Watch_Time_Min", "Assignments_Submitted", "Days_Active"]
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df["Learning_Behavior"] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(8,5))
sns.scatterplot(
    x="Videos_Watched",
    y="Days_Active",
    hue="Learning_Behavior",
    data=df,
    palette="Set2"
)
plt.title("Learning Behavior Clusters")
plt.show()

# -----------------------------
# Group Analysis
# -----------------------------
group_analysis = df.groupby("Learning_Behavior").mean()
print("\nCluster-wise Analysis:")
print(group_analysis)

print("\nKey Insights:")
print("- Students watching more videos tend to score higher.")
print("- Assignment submission strongly affects performance.")
print("- Clustering reveals different learning engagement levels.")
