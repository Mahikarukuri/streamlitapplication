# app_decision.py
# Streamlit app: choose dataset + classifier, show accuracy,
# show a universal visualization (PCA 2D decision regions) where BOTH train and test points
# are colored by their TRUE class label (hue), plus a tree diagram for Decision Tree.

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Classifier Visual Playground", layout="wide")
st.title("🔮 Classifier Playground (All Models Visualized)")
st.write(
    "Pick a dataset and a model. Every model gets a **visual prediction map** "
    "using **PCA to 2D** (decision regions). Train/test points are colored by their TRUE label."
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("1) Dataset + Model")

dataset_name = st.sidebar.selectbox(
    "Choose dataset",
    ["Iris (flowers)", "Wine (chemistry)", "Breast Cancer (diagnosis)"]
)

model_name = st.sidebar.selectbox(
    "Choose model",
    [
        "Decision Tree",
        "Random Forest",
        "Logistic Regression",
        "K-Nearest Neighbors (KNN)",
        "Support Vector Machine (SVM)",
        "Gradient Boosting",
        "Naive Bayes (Gaussian)",
    ],
)

st.sidebar.markdown("---")
st.sidebar.header("2) Train/Test settings")
test_size = st.sidebar.slider("test_size", 0.1, 0.5, 0.25)
random_state = st.sidebar.number_input("random_state", 0, 999999, 42, 1)

st.sidebar.markdown("---")
st.sidebar.header("3) Model hyperparameters")

# Tree / Forest
max_depth = st.sidebar.slider("max_depth (tree/forest)", 1, 20, 4)
min_samples_split = st.sidebar.slider("min_samples_split (tree/forest)", 2, 30, 2)
criterion = st.sidebar.selectbox("criterion (tree)", ["gini", "entropy", "log_loss"])

# Logistic Regression
logreg_c = st.sidebar.slider("C (Logistic Regression)", 0.01, 10.0, 1.0)

# KNN
knn_k = st.sidebar.slider("K (KNN neighbors)", 1, 30, 7)

# SVM
svm_c = st.sidebar.slider("C (SVM)", 0.01, 10.0, 1.0)
svm_kernel = st.sidebar.selectbox("SVM kernel", ["rbf", "linear", "poly", "sigmoid"])

# Gradient Boosting
gb_n_estimators = st.sidebar.slider("n_estimators (GB)", 20, 500, 150, 10)
gb_learning_rate = st.sidebar.slider("learning_rate (GB)", 0.01, 1.0, 0.1)

# Visual settings
st.sidebar.markdown("---")
st.sidebar.header("4) Visualization settings")
mesh_step = st.sidebar.slider("Decision map detail (smaller = sharper, slower)", 0.02, 0.2, 0.06, 0.01)

# Scaling
use_scaling = st.sidebar.checkbox("Use Standard Scaling (recommended)", value=True)

# ----------------------------
# Load dataset
# ----------------------------
def load_dataset(name: str):
    if name.startswith("Iris"):
        return load_iris()
    if name.startswith("Wine"):
        return load_wine()
    return load_breast_cancer()

data = load_dataset(dataset_name)
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
target_names = getattr(data, "target_names", None)

# ----------------------------
# Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=float(test_size),
    random_state=int(random_state),
    stratify=y
)

# ----------------------------
# Build model
# ----------------------------
def build_model(name: str):
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            criterion=criterion,
            random_state=int(random_state),
        )

    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            random_state=int(random_state),
        )

    if name == "Logistic Regression":
        return LogisticRegression(C=float(logreg_c), max_iter=3000)

    if name == "K-Nearest Neighbors (KNN)":
        return KNeighborsClassifier(n_neighbors=int(knn_k))

    if name == "Support Vector Machine (SVM)":
        return SVC(C=float(svm_c), kernel=svm_kernel, probability=True, random_state=int(random_state))

    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=int(gb_n_estimators),
            learning_rate=float(gb_learning_rate),
            random_state=int(random_state),
        )

    return GaussianNB()

base_model = build_model(model_name)

# Pipeline with optional scaling for training on original features
if use_scaling and model_name in ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"]:
    model = Pipeline([("scaler", StandardScaler()), ("model", base_model)])
else:
    model = base_model

# Train on full feature space (for real metrics + manual prediction)
model.fit(X_train, y_train)
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

# ----------------------------
# UI: dataset preview + metrics
# ----------------------------
st.subheader("1) Dataset preview")
c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(X.head(10), use_container_width=True)
with c2:
    counts = y.value_counts().sort_index()
    if target_names is not None:
        st.dataframe(
            pd.DataFrame({"class": [target_names[i] for i in counts.index], "count": counts.values}),
            use_container_width=True
        )
    else:
        st.dataframe(counts.reset_index().rename(columns={"index": "class", "target": "count"}))

st.subheader("2) Accuracy")
m1, m2, m3 = st.columns(3)
m1.metric("Train accuracy", f"{train_acc:.3f}")
m2.metric("Test accuracy", f"{test_acc:.3f}")
m3.metric("Model", model_name)

# ----------------------------
# Universal visualization: PCA → 2D decision regions (with TRUE label hues)
# ----------------------------
st.subheader("3) Visualization: Prediction regions + TRUE-label hues (PCA → 2D)")

st.write(
    "We compress the dataset into **2D using PCA**. The background shows what class the model predicts. "
    "Train/Test points are colored by their **true label** (hue)."
)

# Scale + PCA using training data only
scaler_for_vis = StandardScaler() if use_scaling else None
pca = PCA(n_components=2, random_state=int(random_state))

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

if scaler_for_vis is not None:
    X_train_scaled = scaler_for_vis.fit_transform(X_train_np)
    X_test_scaled = scaler_for_vis.transform(X_test_np)
else:
    X_train_scaled = X_train_np
    X_test_scaled = X_test_np

X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d = pca.transform(X_test_scaled)

# Fit a "visual model" in 2D space (for the decision map background)
vis_model = build_model(model_name)
vis_model.fit(X_train_2d, y_train.to_numpy())

# Mesh grid
x_min, x_max = X_train_2d[:, 0].min() - 1.0, X_train_2d[:, 0].max() + 1.0
y_min, y_max = X_train_2d[:, 1].min() - 1.0, X_train_2d[:, 1].max() + 1.0

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, float(mesh_step)),
    np.arange(y_min, y_max, float(mesh_step))
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = vis_model.predict(grid).reshape(xx.shape)

# Plot
classes = np.unique(y.to_numpy())
n_classes = len(classes)

# We'll use Matplotlib's default colormap (no explicit colors),
# but we DO ensure the same normalization is used for train/test.
norm = plt.Normalize(vmin=classes.min(), vmax=classes.max())

fig = plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.25)

# Train points (circles) colored by true label
sc_train = plt.scatter(
    X_train_2d[:, 0], X_train_2d[:, 1],
    c=y_train.to_numpy(),
    norm=norm,
    marker="o",
    alpha=0.9
)

# Test points (triangles) colored by true label
sc_test = plt.scatter(
    X_test_2d[:, 0], X_test_2d[:, 1],
    c=y_test.to_numpy(),
    norm=norm,
    marker="^",
    alpha=0.9
)

plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.title(f"{model_name} decision regions (trained in PCA-2D space)")

# ---- Legend for classes (colors) ----
cmap = sc_train.cmap  # same cmap for both scatters
class_handles = []
class_labels = []

for c in classes:
    color = cmap(norm(c))
    name = target_names[int(c)] if target_names is not None else str(int(c))
    class_handles.append(Line2D([0], [0], marker="s", linestyle="None", markersize=10, color=color))
    class_labels.append(name)

legend1 = plt.legend(class_handles, class_labels, title="True class (color)", loc="upper right")
plt.gca().add_artist(legend1)

# ---- Legend for split markers ----
split_handles = [
    Line2D([0], [0], marker="o", linestyle="None", markersize=9, color="black", label="Train"),
    Line2D([0], [0], marker="^", linestyle="None", markersize=9, color="black", label="Test"),
]
plt.legend(handles=split_handles, title="Split (marker)", loc="lower right")

st.pyplot(fig)

st.caption(
    "Note: This is a **2D projection**. It’s very useful for intuition, even though the real model is trained in higher dimensions."
)

# ----------------------------
# Decision Tree diagram (extra)
# ----------------------------
if model_name == "Decision Tree":
    st.subheader("4) Extra: Full Decision Tree diagram (trained on original features)")

    inner = model.named_steps["model"] if hasattr(model, "named_steps") else model

    fig_tree = plt.figure(figsize=(18, 8))
    plot_tree(
        inner,
        feature_names=list(X.columns),
        class_names=[str(x) for x in (target_names if target_names is not None else np.unique(y))],
        filled=True,
        rounded=True,
        impurity=True,
    )
    st.pyplot(fig_tree)

# ----------------------------
# Manual prediction (real model)
# ----------------------------
st.subheader("5) Try a manual prediction (real model)")

input_cols = st.columns(3)
user_input = {}

for i, feature in enumerate(X.columns):
    col = input_cols[i % 3]
    fmin = float(X[feature].min())
    fmax = float(X[feature].max())
    fmean = float(X[feature].mean())
    step = (fmax - fmin) / 100.0 if (fmax - fmin) > 0 else 0.01
    step = float(step) if step > 0 else 0.01

    user_input[feature] = col.slider(feature, fmin, fmax, fmean, step=step)

user_df = pd.DataFrame([user_input])
pred = model.predict(user_df)[0]
pred_label = target_names[pred] if target_names is not None else str(pred)
st.success(f"Prediction: **{pred_label}**")

st.write("Your input row:")
st.dataframe(user_df, use_container_width=True)

if hasattr(model, "predict_proba"):
    proba = model.predict_proba(user_df)[0]
    proba_df = pd.DataFrame(
        {
            "class": (target_names if target_names is not None else np.unique(y).astype(str)),
            "probability": proba,
        }
    ).sort_values("probability", ascending=False)
    st.write("Prediction probabilities:")
    st.dataframe(proba_df, use_container_width=True)
else:
    st.info("This model doesn't expose predict_proba().")

# ----------------------------
# Feature importance / coefficients
# ----------------------------
st.subheader("6) Feature importance / coefficients (when available)")

def unwrap_estimator(m):
    return m.named_steps["model"] if hasattr(m, "named_steps") else m

inner = unwrap_estimator(model)

if hasattr(inner, "feature_importances_"):
    imp = pd.Series(inner.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_imp = plt.figure()
    plt.bar(imp.index, imp.values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Importance")
    plt.tight_layout()
    st.pyplot(fig_imp)

elif hasattr(inner, "coef_"):
    coef = inner.coef_
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    avg_abs = np.mean(np.abs(coef), axis=0)
    imp = pd.Series(avg_abs, index=X.columns).sort_values(ascending=False)

    fig_imp = plt.figure()
    plt.bar(imp.index, imp.values)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Avg |coef|")
    plt.tight_layout()
    st.pyplot(fig_imp)
    st.caption("For Logistic Regression, this shows average absolute coefficient magnitude.")

else:
    st.info("This model doesn't provide a simple built-in feature importance metric.")
