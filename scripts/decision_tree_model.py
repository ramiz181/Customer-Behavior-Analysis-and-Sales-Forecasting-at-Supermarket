import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# PATHS
# =====================================================

INPUT_PATH = "../outputs/clustering/customer_segments.csv"

OUTPUT_DIR = "../outputs/decision_tree"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("DECISION TREE — CUSTOMER CLASSIFICATION")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print("\nDataset Loaded Successfully")
print(df.head())

# =====================================================
# ENCODE CATEGORICAL VARIABLES
# =====================================================

print("\nEncoding categorical features...")

label_encoder_gender = LabelEncoder()
label_encoder_loyalty = LabelEncoder()
label_encoder_segment = LabelEncoder()

df["gender_encoded"] = label_encoder_gender.fit_transform(
    df["gender"]
)

df["loyalty_encoded"] = label_encoder_loyalty.fit_transform(
    df["loyalty_tier"]
)

df["segment_encoded"] = label_encoder_segment.fit_transform(
    df["customer_segment"]
)

print("✓ Encoding Completed")

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df[[
    "age",
    "gender_encoded",
    "loyalty_encoded",
    "total_spent",
    "purchase_frequency",
    "avg_basket_size"
]]

y = df["segment_encoded"]

# =====================================================
# TRAIN-TEST SPLIT
# =====================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# TRAIN DECISION TREE MODEL
# =====================================================

print("\nTraining Decision Tree Model...")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

print("✓ Model Trained Successfully")

# =====================================================
# MAKE PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# MODEL EVALUATION
# =====================================================

print("\nMODEL PERFORMANCE")

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy Score: {accuracy:.4f}")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nGenerating Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

cm_path = f"{OUTPUT_DIR}/confusion_matrix.png"

plt.savefig(cm_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"✓ Confusion matrix saved:")
print(cm_path)

# =====================================================
# DECISION TREE VISUALIZATION
# =====================================================

print("\nGenerating Decision Tree Diagram...")

plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=label_encoder_segment.classes_,
    filled=True,
    rounded=True,
    fontsize=9
)

tree_path = f"{OUTPUT_DIR}/decision_tree.png"

plt.savefig(tree_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"✓ Decision tree diagram saved:")
print(tree_path)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:\n")

print(importance_df)

# =====================================================
# SAVE FEATURE IMPORTANCE
# =====================================================

importance_path = f"{OUTPUT_DIR}/feature_importance.csv"

importance_df.to_csv(
    importance_path,
    index=False
)

print(f"\n✓ Feature importance saved:")
print(importance_path)

# =====================================================
# BUSINESS INSIGHTS
# =====================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

top_feature = importance_df.iloc[0]["Feature"]

print(
    f"\nMost important factor affecting "
    f"customer classification: {top_feature}"
)

print(
    "\nThe model helps management identify "
    "high-value customers using behavioral "
    "and demographic features."
)

print("\n✓ Decision Tree Classification Completed")
print("=" * 60)