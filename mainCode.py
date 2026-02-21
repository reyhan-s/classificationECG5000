import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer

# ======================================================
# مرحله 1 و 2: بارگذاری و پیش‌پردازش (بدون انتخاب ویژگی نهایی)
# ======================================================
print("=== STAGE 1 & 2: Loading & Preprocessing ===")

# 1. لود داده‌ها
try:
    train_data = pd.read_csv('ECG5000_TRAIN.txt', header=None, sep=r'\s+')
    test_data = pd.read_csv('ECG5000_TEST.txt', header=None, sep=r'\s+')
except:
    try:
        train_data = pd.read_csv('data/ECG5000_TRAIN.txt', header=None, sep=r'\s+')
        test_data = pd.read_csv('data/ECG5000_TEST.txt', header=None, sep=r'\s+')
    except:
        print("Error: Dataset files not found.")
        exit()

y_train = train_data.iloc[:, 0].astype(int).values
y_test = test_data.iloc[:, 0].astype(int).values
X_train = train_data.iloc[:, 1:].values
X_test = test_data.iloc[:, 1:].values

# تبدیل کلاس‌ها
y_train = y_train - 1
y_test = y_test - 1

# 2. مدیریت داده‌های گمشده
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 3. استانداردسازی
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# رسم نمودار پراکندگی (Scatter Plot) - مرحله 2
plt.figure(figsize=(7, 5)) 
scatter = plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='viridis', s=10, alpha=0.6)
plt.title("Stage 2: Scatter Plot (Feat 1 vs 2)")
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.colorbar(scatter, label="Classes")
plt.show()
# --- محاسبه آنتروپی (فقط برای نمایش نمودار - بدون تغییر X_train) ---
print("Calculating Feature Importance (Visualization Only)...")
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
plt.figure(figsize=(10, 3))
plt.bar(range(len(mi_scores)), mi_scores, color='purple')
plt.title("Stage 6: Feature Importance (Mutual Info - Visualization)")
plt.xlabel("Feature Index")
plt.show()

# ======================================================
# مرحله 3: مقایسه مدل‌ها 
print("\n=== STAGE 3: Comparative Analysis (Selection INSIDE Pipeline) ===")
# اینجا انتخاب ویژگی داخل پایپ‌لاین است.

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
}

comparison_results = []

print("Running models with embedded Feature Selection (No Leakage)...")

for name, model in models.items():
    # --- پایپ‌لاین  ---
    # 1. نمونه‌برداری (Oversampling)
    # 2. انتخاب ویژگی (SelectKBest) -> داخل CV اجرا میشه
    # 3. مدل
    pipeline = ImbPipeline([
        ('sampler', RandomOverSampler(random_state=42)), 
        ('selector', SelectKBest(mutual_info_classif, k=30)), 
        ('model', model)
    ])
    
    # آموزش روی داده‌های کامل 
    pipeline.fit(X_train_scaled, y_train)
    y_pred = pipeline.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    
    comparison_results.append({
        "Model": name, "Accuracy": acc, "F1-Score": f1, "Precision": prec, "Recall": rec
    })

# چاپ جدول
results_df = pd.DataFrame(comparison_results).set_index("Model")
print("\n📊 Model Comparison Table:")
print(results_df)
print("-" * 50)

# --- رسم ماتریس‌ها ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Stage 3: Confusion Matrices (Selection Inside Pipeline)', fontsize=16)

for ax, (name, model) in zip(axes, models.items()):
    pipeline = ImbPipeline([
        ('sampler', RandomOverSampler(random_state=42)),
        ('selector', SelectKBest(mutual_info_classif, k=30)),
        ('model', model)
    ])
    pipeline.fit(X_train_scaled, y_train)
    y_pred = pipeline.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{name} (Acc: {accuracy_score(y_test, y_pred):.2f})")

plt.tight_layout()
plt.show()

# --- رسم ROC ---
y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
n_classes = y_train_bin.shape[1]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Stage 3: ROC Curves (Selection Inside Pipeline)', fontsize=16)

for ax, (name, model) in zip(axes, models.items()):
    classifier = OneVsRestClassifier(
        ImbPipeline([
            ('sampler', RandomOverSampler(random_state=42)),
            ('selector', SelectKBest(mutual_info_classif, k=30)), 
            ('model', model)
        ])
    )
    try:
        y_score = classifier.fit(X_train_scaled, y_train_bin).predict_proba(X_test_scaled)
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=1.5, label=f'Class {i} (AUC={roc_auc[i]:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(f"ROC: {name}")
        ax.legend(loc="lower right", fontsize='small')
    except:
        pass

plt.tight_layout()
plt.show()

# ======================================================
# مرحله 4: تحلیل Overfitting (Pipeline Learning Curve)
# ======================================================
print("\n=== STAGE 4: Overfitting Analysis (Valid Pipeline) ===")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Stage 4: Learning Curves (No Leakage)', fontsize=16)

for ax, (name, model) in zip(axes, models.items()):
    pipeline = ImbPipeline([
        ('sampler', RandomOverSampler(random_state=42)),
        ('selector', SelectKBest(mutual_info_classif, k=30)), 
        ('model', model)
    ])
    
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline, X_train_scaled, y_train, cv=3, scoring="accuracy", n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        ax.plot(train_sizes, train_mean, 'o-', color="r", label="Training")
        ax.plot(train_sizes, test_mean, 'o-', color="g", label="Validation")
        ax.set_title(f"LC: {name}")
        ax.legend(loc="best")
        ax.grid(True)
    except:
        pass

plt.tight_layout()
plt.show()

# ======================================================
# مرحله 5: Regularization (Lasso)
# ======================================================
print("\n=== STAGE 5: Regularization Check (Lasso) ===")
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
feats_kept = np.sum(lasso.coef_ != 0)
print(f"Lasso kept {feats_kept} features.")
# ======================================================
# مرحله 8: مقایسه نهایی و انتخاب مدل برتر (The Final Showdown)
# ======================================================
print("\n=== STAGE 8: Final Optimization & Comparison (All Models) ===")
print("Optimizing all models to find the absolute winner...")

# تعریف فضای جستجو برای هر سه مدل
model_configs = {
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance']
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42, class_weight='balanced'),
        "params": {
            'model__C': [1, 10, 50],
            'model__gamma': ['scale', 0.1],
            'model__kernel': ['rbf']
        }
    },
    "LogReg": {
        "model": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        "params": {
            'model__C': [0.1, 1, 10],
            'model__solver': ['lbfgs']
        }
    }
}

final_results = []
best_estimators = {}

# حلقه بهینه‌سازی برای هر مدل
for name, config in model_configs.items():
    print(f"--> Tuning {name}...")
    
    # پایپ‌لاین استاندارد (بدون نشت اطلاعات)
    pipeline = ImbPipeline([
        ('sampler', RandomOverSampler(random_state=42)),
        ('selector', SelectKBest(mutual_info_classif, k=30)),  
        ('model', config['model'])
    ])
    
    # جستجوی بهترین پارامترها
    grid = GridSearchCV(
        pipeline, 
        config['params'], 
        cv=3, 
        scoring='balanced_accuracy', 
        n_jobs=-1
    )
    
    grid.fit(X_train_scaled, y_train)
    
    # ذخیره بهترین مدل
    best_model = grid.best_estimator_
    best_estimators[name] = best_model
    
    # تست نهایی
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    final_results.append({
        "Model": name,
        "Test Accuracy": acc,
        "Best CV Score (Balanced)": grid.best_score_,
        "Best Params": grid.best_params_
    })

# نمایش جدول رده‌بندی نهایی
results_df = pd.DataFrame(final_results).set_index("Model")
results_df = results_df.sort_values(by="Test Accuracy", ascending=False)
print("\n🏆 FINAL LEADERBOARD (Optimized Models):")
print(results_df)

# --- رسم ماتریس‌های نهایی (سبز) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Stage 8: Final Optimized Confusion Matrices', fontsize=16)

for ax, (name, model) in zip(axes, best_estimators.items()):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    # رنگ سبز برای نشان دادن مدل نهایی
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Greens", cbar=False, ax=ax)
    ax.set_title(f"Optimized {name}\nAcc: {accuracy_score(y_test, y_pred):.2f}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

plt.tight_layout()
plt.show()

# --- چاپ گزارش برنده نهایی ---
winner_name = results_df.index[0] # مدل اول لیست (بیشترین دقت)
print(f"\n✅ CONCLUSION: The Best Model is {winner_name} with Accuracy {results_df.iloc[0]['Test Accuracy']:.4f}")
print("This model is selected for final deployment.")
