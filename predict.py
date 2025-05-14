import deepchem as dc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier

# Load Tox21 dataset for GCN
gcn_featurizer = dc.feat.ConvMolFeaturizer()
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer=gcn_featurizer)
train_dataset, valid_dataset, test_dataset = tox21_datasets

### 1️⃣ Train Graph Convolutional Network (GCN)
gcn_model = dc.models.GraphConvModel(len(tox21_tasks), mode="classification")
gcn_model.fit(train_dataset, nb_epoch=10)
gcn_preds = gcn_model.predict(test_dataset)

### 2️⃣ Load Tox21 dataset for Transformer Model (ECFP Featurization)
transformer_featurizer = dc.feat.CircularFingerprint(size=1024)
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer=transformer_featurizer)
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Ensure valid n_features
n_features = train_dataset.get_data_shape()[0] if isinstance(train_dataset.get_data_shape(), tuple) and len(train_dataset.get_data_shape()) > 0 else 1024

# Define and train Transformer model
transformer_model = dc.models.RobustMultitaskClassifier(len(tox21_tasks), n_features=n_features, mode="classification")
transformer_model.fit(train_dataset, nb_epoch=10)
transformer_preds = transformer_model.predict(test_dataset)

### 3️⃣ Train XGBoost Model
# Convert datasets to NumPy arrays
X_train = train_dataset.X
y_train = train_dataset.y[:, 0]  # Using first task for simplicity

X_test = test_dataset.X

# Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predict with XGBoost
xgb_preds_expanded = np.tile(xgb_preds[:, np.newaxis], (1, 12))

### 4️⃣ Ensemble Predictions (Weighted Average)
weights = [0.6, 0.3, 0.1]  # GCN, Transformer, XGBoost
ensemble_preds = (weights[0] * gcn_preds[..., 1] +
                  weights[1] * transformer_preds[..., 1] +
                  weights[2] * xgb_preds_expanded)
# Get ground truth labels
test_y = test_dataset.y
test_w = test_dataset.w

# Compute ROC curve
fpr, tpr, _ = roc_curve(np.ravel(test_y), np.ravel(ensemble_preds), sample_weight=np.ravel(test_w))
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"Ensemble ROC (area = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Ensemble ROC Curve for Tox21 Prediction")
plt.legend(loc="lower right")
plt.show()
