import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn import svm
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import joblib

df = pd.read_excel('F:/SVM-PCA(1).xlsx', sheet_name='Sheet1')
df.columns = ['DrugName', 'InhibitionRate1', 'InhibitionRate2', 'ID']
df['MergedDrugName'] = df['DrugName']
df.loc[df['ID'].isin([4, 5, 6, 7]), 'MergedDrugName'] = 'Thyroid'

X_raw_global = df[['InhibitionRate1', 'InhibitionRate2']].values
y_global = df['MergedDrugName'].values
le_global = LabelEncoder()
y_global_encoded = le_global.fit_transform(y_global)
X_train_g_raw, X_test_g_raw, y_train_g, y_test_g = train_test_split(
    X_raw_global, y_global_encoded, test_size=0.2, stratify=y_global_encoded, random_state=42)
scaler_global = StandardScaler()
X_train_g_scaled = scaler_global.fit_transform(X_train_g_raw)
X_test_g_scaled = scaler_global.transform(X_test_g_raw)

global_clf = OneVsRestClassifier(
    svm.SVC(kernel='rbf', C=30, gamma=0.1, probability=True, random_state=42))
global_clf.fit(X_train_g_scaled, y_train_g)

x_min, x_max = X_raw_global[:, 0].min() - 1, X_raw_global[:, 0].max() + 1
y_min, y_max = X_raw_global[:, 1].min() - 1, X_raw_global[:, 1].max() + 1
grid_resolution = 300
xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                     np.linspace(y_min, y_max, grid_resolution))
grid_points_raw = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = scaler_global.transform(grid_points_raw)
Z_global = global_clf.predict(grid_points_scaled).reshape(xx.shape)
thyroid_global_label = le_global.transform(["Thyroid"])[0]

df_thyroid = df[df['ID'].isin([4, 5, 6, 7])].copy()
X_raw_thy = df_thyroid[['InhibitionRate1', 'InhibitionRate2']].values
y_thy = df_thyroid['DrugName'].values
le_thy = LabelEncoder()
y_thy_encoded = le_thy.fit_transform(y_thy)
X_train_thy_raw, X_test_thy_raw, y_train_thy, y_test_thy = train_test_split(
    X_raw_thy, y_thy_encoded, test_size=0.2, stratify=y_thy_encoded, random_state=42)
scaler_thy = StandardScaler()
X_train_thy_scaled = scaler_thy.fit_transform(X_train_thy_raw)
X_test_thy_scaled = scaler_thy.transform(X_test_thy_raw)

thyroid_clf = OneVsRestClassifier(
    svm.SVC(kernel='rbf', C=10, gamma=0.1, probability=True, random_state=42))
thyroid_clf.fit(X_train_thy_scaled, y_train_thy)

Z_thyroid = np.full(Z_global.shape, np.nan)
thyroid_mask = (Z_global == thyroid_global_label)
if np.any(thyroid_mask):
    grid_points_thy_masked_raw = grid_points_raw[thyroid_mask.ravel()]
    grid_points_thy_masked_scaled = scaler_thy.transform(grid_points_thy_masked_raw)
    thyroid_preds = thyroid_clf.predict(grid_points_thy_masked_scaled)
    Z_thyroid.ravel()[thyroid_mask.ravel()] = thyroid_preds

fig, ax = plt.subplots(figsize=(15, 10))
contour_global = ax.contourf(xx, yy, Z_global, alpha=0.4, cmap='tab20')
ax.scatter(X_raw_global[:, 0], X_raw_global[:, 1],
           c=y_global_encoded, s=50, edgecolor='k', cmap='tab20')
Z_thyroid_masked = np.ma.masked_where(np.isnan(Z_thyroid), Z_thyroid)
contour_thyroid = ax.contourf(xx, yy, Z_thyroid_masked, alpha=0.5, cmap='Set1')
ax.scatter(X_raw_thy[:, 0], X_raw_thy[:, 1],
           c=y_thy_encoded, s=80, edgecolor='k', cmap='Set1', marker='o')
ax.grid(True, linestyle='--', alpha=0.6)
cbar_global = fig.colorbar(contour_global, ax=ax, ticks=np.arange(len(le_global.classes_)))
cbar_global.set_ticklabels(le_global.classes_)
norm_thyroid = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)
sm_thy = mpl.cm.ScalarMappable(norm=norm_thyroid, cmap='Set1')
sm_thy.set_array([])
cbar_thy = fig.colorbar(sm_thy, ax=ax)
cbar_thy.set_ticks([0, 1, 2, 3])
cbar_thy.set_ticklabels(le_thy.classes_)
fig.tight_layout()
plt.savefig('Global_Thyroid_FourColors.png', dpi=300)
plt.show()

X_raw_14 = df[['InhibitionRate1', 'InhibitionRate2']].values
y_14 = df['DrugName'].values
le_14 = LabelEncoder()
y_14_encoded = le_14.fit_transform(y_14)
X_train_14_raw, X_test_14_raw, y_train_14, y_test_14 = train_test_split(
    X_raw_14, y_14_encoded, test_size=0.2, stratify=y_14_encoded, random_state=42)
scaler_14 = StandardScaler()
X_train_14_scaled = scaler_14.fit_transform(X_train_14_raw)
X_test_14_scaled = scaler_14.transform(X_test_14_raw)

final_clf = OneVsRestClassifier(
    svm.SVC(kernel='rbf', C=10, gamma=0.1, probability=True, random_state=42))
final_clf.fit(X_train_14_scaled, y_train_14)
y_pred_test_14 = final_clf.predict(X_test_14_scaled)

accuracy = accuracy_score(y_test_14, y_pred_test_14)
precision = precision_score(y_test_14, y_pred_test_14, average='weighted')
recall = recall_score(y_test_14, y_pred_test_14, average='weighted')
f1 = f1_score(y_test_14, y_pred_test_14, average='weighted')

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
})
metrics_df.to_csv('14class_SVM_Evaluation_Metrics.csv', index=False)
