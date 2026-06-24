"""
Tabular data-augmentation comparison (dissertation section 3.4.2).

Compares Baseline, Random Oversampling, SMOTE, ADASYN and a GAN on the
NY Bus Breakdown classification task (Breakdown vs Running Late) using only
context features (no target leakage). SMOTE/ADASYN/Random Oversampling are
implemented in pure NumPy + scikit-learn; only the GAN needs TensorFlow.
"""
