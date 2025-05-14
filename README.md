 Ensemble Toxicity Prediction on Tox21 Dataset

This is a deep learning and machine learning-based project for predicting molecular toxicity using multiple models on the [Tox21 dataset](https://tripod.nih.gov/tox21/). This project combines:
- A Graph Convolutional Network (GCN),
- A Transformer-like Robust Multitask Classifier (using ECFP features),
- An XGBoost classifier,
- A weighted ensemble strategy.

  Features
- Multi-task classification on 12 toxicity targets
- Molecular graph representation via GCN
- Fingerprint-based Transformer model
- Gradient boosting with XGBoost
- Weighted ensemble for improved accuracy
- ROC curve and AUC evaluation
  
Requirements
Install all dependencies via:

pip install deepchem tensorflow xgboost scikit-learn matplotlib
⚠️ DeepChem may require additional system packages (e.g. RDKit). Refer to DeepChem installation guide if you encounter issues.

1. Graph Convolutional Network (GCN)
Featurizer: ConvMolFeaturizer
Framework: DeepChem
Epochs: 10
2. Transformer-style Robust Multitask Classifier
Featurizer: CircularFingerprint (ECFP)
Model: RobustMultitaskClassifier
Epochs: 10
3. XGBoost
Using same ECFP features
Task: First task of the Tox21 dataset (binary classification)

⚖️ Ensemble Strategy

The predictions from the three models are combined using a weighted average:

ensemble_preds = (0.6 * gcn_preds + 0.3 * transformer_preds + 0.1 * xgboost_preds)
This results in a smoother and more robust prediction score across tasks.

📈 Evaluation

The final performance is evaluated using the ROC-AUC score and visualized using a ROC curve

Run the Project

python etoxpred_ensemble.py

