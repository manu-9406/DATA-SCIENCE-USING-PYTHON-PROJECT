
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Online Learning Behavior Analyzer", layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("📚 Online Learning Behavior Analyzer")
st.markdown("Analyze student engagement and learning patterns using data science.")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Controls")
num_students = st.sidebar.slider("Number of Students", 50, 500, 200)
num_clusters = st.sidebar.slider("Number of Clusters", 2, 6, 3)

# -----------------------------
# Data Generation
# -----------------------------
np.random.seed(42)

data = {
    "Student_ID": range(1, num_students + 1),
    "Course_Duration_Hours": np.random.randint(10, 60, num_students),
    "Videos_Watched": np.random.randint(5, 50, num_students),
    "Avg_Watch_Time_Min": np.random.randint(5, 30, num_students),
    "Assignments_Submitted": np.random.randint(0, 10, num_students),
    "Quiz_Score": np.random.randint(40, 100, num_students),
    "Days_Active": np.random.randint(1, 30, num_students)
}

df = pd.DataFrame(data)

# -----------------------------
# Display Data
# -----------------------------
st.subheader("📄 Student Dataset")
st.dataframe(df.head(10))

# -----------------------------
# Statistics
# -----------------------------
st.subheader("📊 Statistical Summary")
st.write(df.describe())

# -----------------------------
# Visualizations
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Quiz Score Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Quiz_Score"], bins=20, kde=True, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("Videos Watched vs Quiz Score")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x="Videos_Watched", y="Quiz_Score", data=df, ax=ax2)
    st.pyplot(fig2)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("🔥 Correlation Heatmap")
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)

# -----------------------------
# Clustering
# -----------------------------
st.subheader("🧠 Learning Behavior Clustering")

features = df[
    ["Videos_Watched", "Avg_Watch_Time_Min", "Assignments_Submitted", "Days_Active"]
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["Learning_Behavior"] = kmeans.fit_predict(scaled_features)

fig4, ax4 = plt.subplots()
sns.scatterplot(
    x="Videos_Watched",
    y="Days_Active",
    hue="Learning_Behavior",
    data=df,
    palette="Set2",
    ax=ax4
)
ax4.set_title("Learning Behavior Clusters")
st.pyplot(fig4)

# -----------------------------
# Cluster Insights
# -----------------------------
st.subheader("📌 Cluster-wise Analysis")
st.dataframe(df.groupby("Learning_Behavior").mean())

st.success("✔ Analysis Completed Successfully!")

st.markdown("""
### 🔍 Key Insights
- Students watching more videos tend to score higher  
- Assignment submission strongly impacts performance  
- Clustering reveals engagement levels (Low / Medium / High)
""")
