import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================
# PATHS
# =====================================================

INPUT_PATH = "../data/features/customer_features.csv"

OUTPUT_DIR = "../outputs/clustering"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("CUSTOMER SEGMENTATION — K-MEANS CLUSTERING")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print("\nDataset Loaded Successfully")
print(df.head())

# =====================================================
# SELECT FEATURES
# =====================================================

features = df[[
    "total_spent",
    "purchase_frequency",
    "avg_basket_size"
]]

# =====================================================
# NORMALIZE FEATURES
# =====================================================

print("\nScaling Features...")

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

# =====================================================
# APPLY K-MEANS
# =====================================================

print("\nApplying K-Means Clustering...")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(scaled_features)

print("✓ Clustering Completed")

# =====================================================
# CLUSTER ANALYSIS
# =====================================================

print("\nCluster Summary:\n")

cluster_summary = df.groupby("cluster")[[
    "total_spent",
    "purchase_frequency",
    "avg_basket_size"
]].mean()

print(cluster_summary)

# =====================================================
# LABEL CLUSTERS
# =====================================================

cluster_labels = {
    0: "Low Value",
    1: "Mid Value",
    2: "High Value"
}

df["customer_segment"] = df["cluster"].map(cluster_labels)

# =====================================================
# SAVE CLUSTERED DATA
# =====================================================

output_csv = f"{OUTPUT_DIR}/customer_segments.csv"

df.to_csv(output_csv, index=False)

print(f"\n✓ Clustered data saved:")
print(output_csv)

# =====================================================
# VISUALIZATION
# =====================================================

print("\nGenerating Scatter Plot...")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))

scatter = sns.scatterplot(
    data=df,
    x="purchase_frequency",
    y="total_spent",
    hue="customer_segment",
    palette="Set2",
    s=100
)

plt.title(
    "Customer Segmentation using K-Means",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Purchase Frequency")
plt.ylabel("Total Amount Spent")

plt.legend(title="Customer Segment")

plot_path = f"{OUTPUT_DIR}/customer_clusters.png"

plt.savefig(plot_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"✓ Scatter plot saved:")
print(plot_path)

# =====================================================
# BUSINESS INSIGHTS
# =====================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

segment_counts = df["customer_segment"].value_counts()

for segment, count in segment_counts.items():
    print(f"{segment:<15} : {count} customers")

high_value = df[df["customer_segment"] == "High Value"]

high_value_revenue = high_value["total_spent"].sum()

total_revenue = df["total_spent"].sum()

contribution = (
    high_value_revenue / total_revenue
) * 100

print(
    f"\nHigh Value Customers contribute "
    f"{contribution:.2f}% of total revenue."
)

print("\n✓ K-Means Clustering Completed Successfully")
print("=" * 60)