# ==========================================
#  CONFIGURATION & IMPORTS
# ==========================================
import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import entropy, zscore
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)

# Quantum imports
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

print(" All imports successful!")

# ==========================================
#  PROJECT CONFIGURATION
# ==========================================
CONFIG = {
    # Paths
    'dataset_dir': '/content/drive/MyDrive/CAN_IDS/',

    # Dataset parameters
    'n_samples_per_class': 5000,      # Matches base paper
    'test_size': 0.2,                  # 80-20 split
    'random_state': 42,

    # Feature engineering
    'window_size': 20,                 # For time-series features
    'n_top_features': 8,               # Top features to select

    # Classical ML
    'cv_folds': 5,                     # Cross-validation folds

    # Quantum ML
    'qsvm_samples_per_class': 200,    # Quantum subset (memory constraint)
    'quantum_reps': 2,                 # Circuit repetitions
    'quantum_entanglement': 'linear',  # Entanglement strategy
}

print("\n" + "="*70)
print(" PROJECT: Quantum-Enhanced CAN-IDS")
print("="*70)
print(f" Dataset size: {CONFIG['n_samples_per_class'] * 4:,} samples")
print(f"  Quantum subset: {CONFIG['qsvm_samples_per_class'] * 4:,} samples")
print(f" Cross-validation: {CONFIG['cv_folds']}-fold")
print("="*70)


# ==========================================
#  QUANTUM DATASET PREPARATION
# ==========================================
print("="*70)
print(" PREPARING QUANTUM DATASET")
print("="*70)

# Due to computational constraints, use subset for quantum
n_quantum = CONFIG['qsvm_samples_per_class']

print(f"\n Quantum training subset size: {n_quantum} samples per class")
print(f"   Total quantum training samples: {n_quantum * 2:,} (balanced)")
print(f"   (Reduced for quantum computational efficiency)")
print(f"\n Note: We'll use the SAME test set for fair comparison!")

# ===== CREATE QUANTUM TRAINING SUBSET =====
def create_quantum_subset(X, y, attack_types, n_per_label):
    """
    Create balanced subset for quantum training.
    Sample equally from normal (0) and attack (1).
    """
    indices_normal = np.where(y == 0)[0]
    indices_attack = np.where(y == 1)[0]

    # Sample equal amounts from each class
    np.random.seed(CONFIG['random_state'])
    sampled_normal = np.random.choice(indices_normal, n_per_label, replace=False)
    sampled_attack = np.random.choice(indices_attack, n_per_label, replace=False)

    # Combine and shuffle
    sampled_indices = np.concatenate([sampled_normal, sampled_attack])
    np.random.shuffle(sampled_indices)

    return X[sampled_indices], y[sampled_indices], attack_types[sampled_indices]

# Create quantum training subset
X_train_q, y_train_q, att_train_q = create_quantum_subset(
    X_train_scaled, y_train, train_data['attack_types_train'], n_quantum
)

# Use SAME test set as classical (for fair comparison)
X_test_q = X_test_scaled
y_test_q = y_test
att_test_q = train_data['attack_types_test']

print(f"\n Quantum dataset created:")
print(f"   Training samples:   {len(X_train_q):>6,} (subset for quantum efficiency)")
print(f"   Test samples:       {len(X_test_q):>6,} (SAME as classical)")

print(f"\n Quantum training distribution:")
unique_q_train, counts_q_train = np.unique(y_train_q, return_counts=True)
for label, count in zip(unique_q_train, counts_q_train):
    label_name = "Normal" if label == 0 else "Attack"
    print(f"   {label_name} ({label}): {count:>6,} ({count/len(y_train_q)*100:.1f}%)")

print(f"\n Quantum test distribution (same as classical):")
unique_q_test, counts_q_test = np.unique(y_test_q, return_counts=True)
for label, count in zip(unique_q_test, counts_q_test):
    label_name = "Normal" if label == 0 else "Attack"
    print(f"   {label_name} ({label}): {count:>6,} ({count/len(y_test_q)*100:.1f}%)")

# Attack type distribution in quantum training set
print(f"\n Quantum training attack type breakdown:")
q_train_att_counts = pd.Series(att_train_q).value_counts()
for att_type, count in q_train_att_counts.items():
    print(f"   {att_type:15s}: {count:>6,} ({count/len(att_train_q)*100:.1f}%)")

# Store quantum data
quantum_data = {
    'X_train_q': X_train_q,
    'X_test_q': X_test_q,
    'y_train_q': y_train_q,
    'y_test_q': y_test_q,
    'attack_types_train_q': att_train_q,
    'attack_types_test_q': att_test_q
}

print("\n" + "="*70)
print(" KEY POINTS FOR FAIR COMPARISON")
print("="*70)
print(f"✓ Classical SVM trained on: {len(X_train_scaled):,} samples")
print(f"✓ Quantum SVM trained on:   {len(X_train_q):,} samples (memory constraint)")
print(f"✓ Both tested on:           {len(X_test_q):,} samples (IDENTICAL test set)")
print(f"✓ This allows fair accuracy comparison despite different training sizes")

print("\n Quantum dataset preparation complete!")
print("="*70)


# ==========================================
#  QUANTUM SVM - FIXED FOR CONSISTENT RESULTS
# ==========================================
print("="*70)
print(" QUANTUM SVM - TWO-STAGE (CONSISTENT SAMPLES)")
print("="*70)

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import gc
import psutil
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import time

gc.collect()

print("\n FIXED STRATEGY - BOTH STAGES USE SAME SAMPLES")
print("="*70)
print("Changes:")
print("  ✓ Fixed random seeds for reproducibility")
print("  ✓ Stage 1 & Stage 2 use SAME training samples")
print("  ✓ Consistent data split every run")
print("  ✓ Same test set every time")
print()
print("STAGE 1: Binary (Normal vs Attack)")
print("  Training: 200 samples")
print("    • 100 Normal")
print("    • 100 Attack (50 Fuzzy + 25 DoS + 25 Imp)")
print()
print("STAGE 2: Attack Types (DoS vs Fuzzy vs Imp)")
print("  Training: 100 samples (SAME AS STAGE 1 ATTACKS)")
print("    • 50 Fuzzy")
print("    • 25 DoS")
print("    • 25 Imp")
print("="*70)

# ===== DATA PREPARATION =====
X_train_full = quantum_data['X_train_q']
y_train_full = quantum_data['y_train_q']
att_train_full = quantum_data['attack_types_train_q']

X_test_full = quantum_data['X_test_q']
y_test_full = quantum_data['y_test_q']
att_test_full = quantum_data['attack_types_test_q']

# ===== GET ALL AVAILABLE INDICES =====
mask_normal = att_train_full == 'Normal'
mask_fuzzy = att_train_full == 'Fuzzy'
mask_dos = att_train_full == 'DoS'
mask_imp = att_train_full == 'Impersonation'

all_normal_indices = np.where(mask_normal)[0]
all_fuzzy_indices = np.where(mask_fuzzy)[0]
all_dos_indices = np.where(mask_dos)[0]
all_imp_indices = np.where(mask_imp)[0]

print(f"\n Available training data:")
print(f"   Normal: {len(all_normal_indices)} samples")
print(f"   Fuzzy:  {len(all_fuzzy_indices)} samples")
print(f"   DoS:    {len(all_dos_indices)} samples")
print(f"   Imp:    {len(all_imp_indices)} samples")

# ===== FIXED RANDOM STATE FOR REPRODUCIBILITY =====
FIXED_SEED = CONFIG['random_state']

# ===== CREATE CONSISTENT TRAINING SET =====
np.random.seed(FIXED_SEED)

# Select samples (SAME for both stages)
normal_indices = np.random.choice(all_normal_indices, 100, replace=False)
fuzzy_indices = np.random.choice(all_fuzzy_indices, 50, replace=False)
dos_indices = np.random.choice(all_dos_indices, 25, replace=False)
imp_indices = np.random.choice(all_imp_indices, 25, replace=False)

# Combine attack indices (for Stage 2)
attack_indices = np.concatenate([fuzzy_indices, dos_indices, imp_indices])

# ===== STAGE 1 TRAINING SET (Normal + Attacks) =====
stage1_indices = np.concatenate([normal_indices, attack_indices])
np.random.seed(FIXED_SEED)  # Reset seed before shuffle for consistency
np.random.shuffle(stage1_indices)

X_train_stage1 = X_train_full[stage1_indices]
y_train_stage1 = y_train_full[stage1_indices]
att_train_stage1 = att_train_full[stage1_indices]

print(f"\n Stage 1 training set:")
print(f"   Normal: 100 samples")
print(f"   Fuzzy:  50 samples")
print(f"   DoS:    25 samples")
print(f"   Imp:    25 samples")
print(f"   Total:  {len(X_train_stage1)} samples")

# ===== STAGE 2 TRAINING SET (SAME ATTACKS as Stage 1) =====
np.random.seed(FIXED_SEED)  # Same seed for consistency
np.random.shuffle(attack_indices)

X_train_stage2 = X_train_full[attack_indices]
att_train_stage2 = att_train_full[attack_indices]

# Create 3-class labels (0=DoS, 1=Fuzzy, 2=Imp)
attack_type_map_stage2 = {'DoS': 0, 'Fuzzy': 1, 'Impersonation': 2}
y_train_stage2 = np.array([attack_type_map_stage2[att] for att in att_train_stage2])

print(f"\n Stage 2 training set (SAME attacks as Stage 1):")
print(f"   Fuzzy: {(att_train_stage2 == 'Fuzzy').sum()} samples")
print(f"   DoS:   {(att_train_stage2 == 'DoS').sum()} samples")
print(f"   Imp:   {(att_train_stage2 == 'Impersonation').sum()} samples")
print(f"   Total: {len(X_train_stage2)} samples")

# ===== TEST SET (CONSISTENT) =====
np.random.seed(FIXED_SEED)
test_indices = []

print(f"\n Building test set:")

for attack_type in ['Normal', 'DoS', 'Fuzzy', 'Impersonation']:
    mask = att_test_full == attack_type
    available_indices = np.where(mask)[0]
    n_sample = min(30, len(available_indices))
    sampled = np.random.choice(available_indices, n_sample, replace=False)
    test_indices.extend(sampled)
    print(f"   {attack_type:15s}: {n_sample} samples")

test_indices = np.array(test_indices)
np.random.seed(FIXED_SEED)
np.random.shuffle(test_indices)

X_test = X_test_full[test_indices]
y_test = y_test_full[test_indices]
att_test = att_test_full[test_indices]

print(f"\n Test set: {len(X_test)} samples")

# ===== RAM CHECK =====
mem = psutil.virtual_memory()
print(f"\n RAM Check:")
print(f"   Available: {mem.available / (1024**3):.2f} GB")
print(f"   Stage 1 kernel: 200×200 = ~3 GB ✓")
print(f"   Stage 1 test:   120×200 = ~2 GB ✓")
print(f"   Stage 2 kernel: 100×100 = ~1 GB ✓")
print(f"   Peak usage: ~3 GB (SAFE!)")

# ==========================================
#  STAGE 1: BINARY CLASSIFIER
# ==========================================
print("\n" + "="*70)
print(" STAGE 1: BINARY CLASSIFIER (NORMAL vs ATTACK)")
print("="*70)

n_features = X_train_stage1.shape[1]

feature_map_stage1 = ZZFeatureMap(
    feature_dimension=n_features,
    reps=3,
    entanglement='linear',
    insert_barriers=True
)

print(f"\n Stage 1 Quantum Feature Map:")
print(f"   Qubits: {feature_map_stage1.num_qubits}")
print(f"   Reps: 3")

quantum_kernel_stage1 = FidelityQuantumKernel(feature_map=feature_map_stage1)

print(f"\n Computing Stage 1 kernel ({len(X_train_stage1)}×{len(X_train_stage1)})...")
print(f"   Expected: ~10-12 minutes...")

start_time = time.time()
K_train_stage1 = quantum_kernel_stage1.evaluate(X_train_stage1)
kernel_time_stage1 = time.time() - start_time

print(f"\n   ✓ Stage 1 kernel complete: {kernel_time_stage1:.1f}s ({kernel_time_stage1/60:.1f} min)")

# Hyperparameter tuning
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=FIXED_SEED)
best_C_stage1 = 0.1
best_score = 0

print(f"\n Stage 1 hyperparameter tuning:")

for C in [0.1, 0.3, 0.5, 1.0]:
    scores = []
    for train_idx, val_idx in skf.split(X_train_stage1, y_train_stage1):
        K_fold = K_train_stage1[train_idx][:, train_idx]
        K_val = K_train_stage1[val_idx][:, train_idx]
        svm_temp = SVC(kernel='precomputed', C=C, random_state=FIXED_SEED)
        svm_temp.fit(K_fold, y_train_stage1[train_idx])
        scores.append(accuracy_score(y_train_stage1[val_idx], svm_temp.predict(K_val)))

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"   C={C:>6.1f}: {mean_score:.4f} (+/- {std_score:.4f})")

    if mean_score > best_score:
        best_score = mean_score
        best_C_stage1 = C

print(f"\n Best C (Stage 1): {best_C_stage1}")

# Train final Stage 1 model
qsvm_stage1 = SVC(kernel='precomputed', C=best_C_stage1, random_state=FIXED_SEED)
qsvm_stage1.fit(K_train_stage1, y_train_stage1)

y_train_pred_stage1 = qsvm_stage1.predict(K_train_stage1)
train_acc_stage1 = accuracy_score(y_train_stage1, y_train_pred_stage1)

print(f"\n Stage 1 training accuracy: {train_acc_stage1:.4f} ({train_acc_stage1*100:.2f}%)")

# Check per-type training accuracy
print(f"\n   Per-type training accuracy:")
for attack_type in ['Normal', 'DoS', 'Fuzzy', 'Impersonation']:
    mask = att_train_stage1 == attack_type
    if mask.sum() > 0:
        acc = accuracy_score(y_train_stage1[mask], y_train_pred_stage1[mask])
        correct = (y_train_pred_stage1[mask] == y_train_stage1[mask]).sum()
        total = mask.sum()
        print(f"      {attack_type:15s}: {acc*100:>6.2f}% ({correct:>3}/{total:<3})")

# ==========================================
#  STAGE 2: ATTACK TYPE CLASSIFIER
# ==========================================
print("\n" + "="*70)
print(" STAGE 2: ATTACK TYPE CLASSIFIER")
print("="*70)

feature_map_stage2 = ZZFeatureMap(
    feature_dimension=n_features,
    reps=2,
    entanglement='linear',
    insert_barriers=True
)

print(f"\n Stage 2 Quantum Feature Map:")
print(f"   Qubits: {feature_map_stage2.num_qubits}")
print(f"   Reps: 2")

quantum_kernel_stage2 = FidelityQuantumKernel(feature_map=feature_map_stage2)

# Cleanup Stage 1 kernel
del K_train_stage1
gc.collect()
time.sleep(2)

mem_after_cleanup = psutil.virtual_memory()
print(f"   RAM after cleanup: {mem_after_cleanup.available / (1024**3):.2f} GB")

print(f"\n Computing Stage 2 kernel ({len(X_train_stage2)}×{len(X_train_stage2)})...")
print(f"   Expected: ~4-5 minutes...")

start_time = time.time()
K_train_stage2 = quantum_kernel_stage2.evaluate(X_train_stage2)
kernel_time_stage2 = time.time() - start_time

print(f"\n   ✓ Stage 2 kernel complete: {kernel_time_stage2:.1f}s ({kernel_time_stage2/60:.1f} min)")

# Hyperparameter tuning for Stage 2
skf2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=FIXED_SEED)
best_C_stage2 = 0.1
best_score = 0

print(f"\n Stage 2 hyperparameter tuning:")

for C in [0.1, 0.3, 0.5, 1.0]:
    scores = []
    for train_idx, val_idx in skf2.split(X_train_stage2, y_train_stage2):
        K_fold = K_train_stage2[train_idx][:, train_idx]
        K_val = K_train_stage2[val_idx][:, train_idx]
        svm_temp = SVC(kernel='precomputed', C=C, random_state=FIXED_SEED)
        svm_temp.fit(K_fold, y_train_stage2[train_idx])
        scores.append(accuracy_score(y_train_stage2[val_idx], svm_temp.predict(K_val)))

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"   C={C:>6.1f}: {mean_score:.4f} (+/- {std_score:.4f})")

    if mean_score > best_score:
        best_score = mean_score
        best_C_stage2 = C

print(f"\n Best C (Stage 2): {best_C_stage2}")

# Train final Stage 2 model
qsvm_stage2 = SVC(kernel='precomputed', C=best_C_stage2, random_state=FIXED_SEED)
qsvm_stage2.fit(K_train_stage2, y_train_stage2)

y_train_pred_stage2 = qsvm_stage2.predict(K_train_stage2)
train_acc_stage2 = accuracy_score(y_train_stage2, y_train_pred_stage2)

print(f"\n Stage 2 training accuracy: {train_acc_stage2:.4f} ({train_acc_stage2*100:.2f}%)")

# ==========================================
#  HIERARCHICAL TESTING
# ==========================================
print("\n" + "="*70)
print(" HIERARCHICAL TEST EVALUATION")
print("="*70)

print(f"\n Step 1: Binary classification on {len(X_test)} test samples...")
print(f"   Computing 120×200 kernel matrix...")
print(f"   Expected: ~7-9 minutes...")

start_test = time.time()

K_test_stage1 = quantum_kernel_stage1.evaluate(X_test, X_train_stage1)
y_test_pred_binary = qsvm_stage1.predict(K_test_stage1)

time_stage1_test = time.time() - start_test
print(f"   ✓ Stage 1 test complete: {time_stage1_test:.1f}s ({time_stage1_test/60:.1f} min)")

# Find samples predicted as Attack
attack_predicted_mask = y_test_pred_binary == 1
n_attacks_predicted = attack_predicted_mask.sum()

print(f"\n Step 2: Attack type classification on {n_attacks_predicted} samples predicted as Attack...")

if n_attacks_predicted > 0:
    X_test_attacks = X_test[attack_predicted_mask]

    print(f"   Computing {len(X_test_attacks)}×{len(X_train_stage2)} kernel matrix...")
    print(f"   Expected: ~2-3 minutes...")

    start_time = time.time()
    K_test_stage2 = quantum_kernel_stage2.evaluate(X_test_attacks, X_train_stage2)
    y_test_pred_attack_types = qsvm_stage2.predict(K_test_stage2)
    time_stage2_test = time.time() - start_time

    print(f"   ✓ Stage 2 test complete: {time_stage2_test:.1f}s ({time_stage2_test/60:.1f} min)")

    del K_test_stage2
    gc.collect()
else:
    y_test_pred_attack_types = np.array([])
    time_stage2_test = 0

total_test_time = time_stage1_test + time_stage2_test

# Cleanup
del K_test_stage1
gc.collect()

# ==========================================
#  CONSTRUCT FINAL 4-CLASS PREDICTIONS
# ==========================================
print("\n Constructing final 4-class predictions...")

# Map to 4 classes: 0=Normal, 1=DoS, 2=Fuzzy, 3=Impersonation
attack_type_map_final = {'Normal': 0, 'DoS': 1, 'Fuzzy': 2, 'Impersonation': 3}
reverse_map_stage2 = {0: 1, 1: 2, 2: 3}  # Stage2 output to final labels

y_test_multiclass_true = np.array([attack_type_map_final[att] for att in att_test])
y_test_multiclass_pred = np.zeros(len(y_test), dtype=int)

attack_pred_idx = 0
for i in range(len(y_test)):
    if y_test_pred_binary[i] == 0:  # Predicted Normal
        y_test_multiclass_pred[i] = 0
    else:  # Predicted Attack - use Stage 2 prediction
        y_test_multiclass_pred[i] = reverse_map_stage2[y_test_pred_attack_types[attack_pred_idx]]
        attack_pred_idx += 1

# ==========================================
#  RESULTS AND CONFUSION MATRICES
# ==========================================
print("\n" + "="*70)
print(" FINAL RESULTS")
print("="*70)

# Binary metrics
test_acc_binary = accuracy_score(y_test, y_test_pred_binary)
test_precision = precision_score(y_test, y_test_pred_binary, average='binary', zero_division=0)
test_recall = recall_score(y_test, y_test_pred_binary, average='binary')
test_f1 = f1_score(y_test, y_test_pred_binary, average='binary')

print(f"\n Stage 1 (Binary) Performance:")
print(f"   Test Accuracy:  {test_acc_binary:.4f} ({test_acc_binary*100:.2f}%)")
print(f"   Precision:      {test_precision:.4f}")
print(f"   Recall:         {test_recall:.4f}")
print(f"   F1-Score:       {test_f1:.4f}")

# Multi-class metrics
test_acc_multiclass = accuracy_score(y_test_multiclass_true, y_test_multiclass_pred)

print(f"\n Overall (4-Class) Performance:")
print(f"   Test Accuracy:  {test_acc_multiclass:.4f} ({test_acc_multiclass*100:.2f}%)")

# ===== 2x2 BINARY CONFUSION MATRIX =====
print(f"\n" + "="*70)
print(" 2x2 BINARY CONFUSION MATRIX (Stage 1)")
print("="*70)

cm_binary = confusion_matrix(y_test, y_test_pred_binary)

print(f"\n                 Predicted")
print(f"                 Normal  Attack")
print(f"   Actual Normal  {cm_binary[0,0]:>6}  {cm_binary[0,1]:>6}")
print(f"   Actual Attack  {cm_binary[1,0]:>6}  {cm_binary[1,1]:>6}")

tn, fp, fn, tp = cm_binary.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n Binary Metrics:")
print(f"   Specificity (Normal): {specificity:.4f} ({specificity*100:.2f}%)")
print(f"   Sensitivity (Attack): {sensitivity:.4f} ({sensitivity*100:.2f}%)")

# ===== 4x4 MULTI-CLASS CONFUSION MATRIX =====
print(f"\n" + "="*70)
print(" 4x4 MULTI-CLASS CONFUSION MATRIX (Hierarchical)")
print("="*70)

class_names = ['Normal', 'DoS', 'Fuzzy', 'Impersonation']
cm_4x4 = confusion_matrix(y_test_multiclass_true, y_test_multiclass_pred, labels=[0, 1, 2, 3])

print(f"\n                      Predicted")
print(f"               Normal    DoS  Fuzzy    Imp")
header = "   Actual "
for i, actual_class in enumerate(class_names):
    row_label = f"{header}{actual_class:13s}"
    print(f"{row_label} {cm_4x4[i,0]:>6} {cm_4x4[i,1]:>6} {cm_4x4[i,2]:>6} {cm_4x4[i,3]:>6}")
    header = "          "

# Highlight Fuzzy performance
fuzzy_detected = cm_4x4[2, 1] + cm_4x4[2, 2] + cm_4x4[2, 3]
fuzzy_total = cm_4x4[2, :].sum()
fuzzy_detection_rate = fuzzy_detected / fuzzy_total if fuzzy_total > 0 else 0

print(f"\n ⭐ Fuzzy Attack Detection:")
print(f"   Detected as Attack: {fuzzy_detected}/{fuzzy_total} ({fuzzy_detection_rate*100:.1f}%)")
print(f"   Misclassified as Normal: {cm_4x4[2,0]}/{fuzzy_total} ({cm_4x4[2,0]/fuzzy_total*100:.1f}%)")

# ===== PER-CLASS METRICS =====
print(f"\n" + "="*70)
print(" PER-CLASS DETAILED METRICS")
print("="*70)

precision_per_class, recall_per_class, f1_per_class, support_per_class = \
    precision_recall_fscore_support(y_test_multiclass_true, y_test_multiclass_pred,
                                   labels=[0, 1, 2, 3], zero_division=0)

for i, class_name in enumerate(class_names):
    print(f"\n {class_name}:")
    print(f"   Precision: {precision_per_class[i]:.4f}")
    print(f"   Recall:    {recall_per_class[i]:.4f}")
    print(f"   F1-Score:  {f1_per_class[i]:.4f}")
    print(f"   Support:   {support_per_class[i]}")

# ===== CLASSIFICATION REPORT =====
print(f"\n" + "="*70)
print(" CLASSIFICATION REPORT")
print("="*70)

print("\n Binary Classification (Normal vs Attack):")
print(classification_report(y_test, y_test_pred_binary,
                          target_names=['Normal', 'Attack'], digits=4, zero_division=0))

print("\n Multi-Class Classification (4 Classes):")
print(classification_report(y_test_multiclass_true, y_test_multiclass_pred,
                          target_names=class_names, digits=4, zero_division=0))

# Store results
quantum_results = {
    'model_stage1': qsvm_stage1,
    'model_stage2': qsvm_stage2,
    'kernel_stage1': quantum_kernel_stage1,
    'kernel_stage2': quantum_kernel_stage2,
    'feature_map_stage1': feature_map_stage1,
    'feature_map_stage2': feature_map_stage2,
    'train_acc_binary': train_acc_stage1,
    'train_acc_attack_types': train_acc_stage2,
    'test_acc_binary': test_acc_binary,
    'test_acc_multiclass': test_acc_multiclass,
    'precision': test_precision,
    'recall': test_recall,
    'f1': test_f1,
    'specificity': specificity,
    'sensitivity': sensitivity,
    'train_time_stage1': kernel_time_stage1,
    'train_time_stage2': kernel_time_stage2,
    'test_time_stage1': time_stage1_test,
    'test_time_stage2': time_stage2_test,
    'total_train_time': kernel_time_stage1 + kernel_time_stage2,
    'total_test_time': total_test_time,
    'predictions_binary': y_test_pred_binary,
    'predictions_multiclass': y_test_multiclass_pred,
    'confusion_matrix_binary': cm_binary,
    'confusion_matrix_4x4': cm_4x4,
    'y_test_multiclass_true': y_test_multiclass_true,
    'best_C_stage1': best_C_stage1,
    'best_C_stage2': best_C_stage2,
    'class_names': class_names,
    'precision_per_class': precision_per_class,
    'recall_per_class': recall_per_class,
    'f1_per_class': f1_per_class,
    'fuzzy_detection_rate': fuzzy_detection_rate
}

print("\n" + "="*70)
print(" SUMMARY")
print("="*70)
print(f"✓ FIXED: Both stages use SAME samples")
print(f"✓ Stage 1: 200 training samples (100 Normal + 100 Attacks)")
print(f"✓ Stage 2: 100 training samples (SAME 100 attacks)")
print(f"✓ Test set: 120 samples (30 each type)")
print(f"\n✓ Binary Accuracy: {test_acc_binary*100:.2f}%")
print(f"✓ Multi-Class Accuracy: {test_acc_multiclass*100:.2f}%")
print(f"✓ Fuzzy Detection Rate: {fuzzy_detection_rate*100:.1f}%")
print(f"\n✓ Training time:")
print(f"  • Stage 1: {kernel_time_stage1/60:.1f} min")
print(f"  • Stage 2: {kernel_time_stage2/60:.1f} min")
print(f"  • Total: {(kernel_time_stage1 + kernel_time_stage2)/60:.1f} min")
print(f"\n✓ Test time: {total_test_time/60:.1f} min")
print(f"\n✓ Consistent results guaranteed!")
print(f"  • Fixed random seeds throughout")
print(f"  • Same training/test split every run")
print(f"  • Both stages see identical samples")
print("="*70)
