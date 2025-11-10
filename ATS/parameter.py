import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from skimage.feature import hog
from skimage.transform import resize
from tqdm import tqdm
import os
import sys
import string

DATA_FILEPATH = "D:\\UTS Machine Vision\\emnist-letters-train.csv"
SAMPLES_TO_USE = 13000  
N_SPLITS = 5            

# Optimal Parameters from Tuning
OPTIMAL_C = 10
OPTIMAL_GAMMA = 'scale'
KERNEL = 'rbf'

# --- Utility Function: Map 1-26 to A-Z ---
def get_az_labels():
    """Returns a list of character labels 'A' through 'Z'."""
    # EMNIST labels 1=A, 2=B, ..., 26=Z
    return list(string.ascii_uppercase)

# Function to extract HOG features from an image
def extract_hog_features(image):
    """
    Extracts HOG features using the optimal, fixed parameters.
    """
    # Resize and normalize image
    if image.ndim == 3:
        image = image.mean(axis=2)
    if image.dtype != np.uint8 and image.max() > 1.0:
        image = (image / 255.0) 
        
    resized_image = resize(image, (28, 28), anti_aliasing=True)
    
    # HOG Parameters (fixed)
    features = hog(resized_image, 
                   orientations=9, 
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), 
                   visualize=False, 
                   transform_sqrt=True,
                   feature_vector=True)
    return features

# --- Execution Starts Here ---
print("="*80)
print(f"FINAL HOG + SVM CHARACTER CLASSIFICATION ANALYSIS ({SAMPLES_TO_USE} Samples)")
print(f"SVM Parameters: C={OPTIMAL_C}, gamma={OPTIMAL_GAMMA}, kernel={KERNEL}")
print("="*80)

# --- 1. Load and Preprocess Data ---
try:
    print(f"1. Loading data from: {DATA_FILEPATH}")
    if not os.path.exists(DATA_FILEPATH):
        raise FileNotFoundError(f"File not found at: {DATA_FILEPATH}")
        
    data = pd.read_csv(DATA_FILEPATH, nrows=SAMPLES_TO_USE)
    
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

# X is pixel data, y is labels (1-26)
X_full = data.iloc[:, 1:].values.astype('float32')
y_full = data.iloc[:, 0].values.astype('int')
X_images_full = X_full.reshape(-1, 28, 28)

print(f"   -> Successfully loaded {X_full.shape[0]} samples (26 classes).")

# --- 2. HOG Feature Extraction ---
print("\n2. Extracting HOG features for all samples...")
X_hog_full = []
# Ensure this extraction is done only once before CV
for img in tqdm(X_images_full, desc="HOG Feature Extraction"):
    X_hog_full.append(extract_hog_features(img))
X_hog_full = np.array(X_hog_full)
print(f"   -> Final HOG features shape: {X_hog_full.shape}")

# --- 3. Cross-Validation (LOOCV Proxy) Setup ---
print("\n" + "-"*80)
print("⚠️ LOOCV WARNING: True LOOCV on 13,000 samples is too slow.")
print(f"Using Stratified K-Fold Cross-Validation (K={N_SPLITS}) to estimate performance.")
print("ESTIMATED RUN TIME: 1 to 3+ hours, depending on CPU resources.")
print("-"*80)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
# Instantiate the SVM with optimal parameters
svm_classifier = SVC(C=OPTIMAL_C, 
                     gamma=OPTIMAL_GAMMA, 
                     kernel=KERNEL, 
                     random_state=42, 
                     verbose=False)

accuracy_scores = []
y_true_all = []
y_pred_all = []
fold_count = 1

# --- 4. K-Fold CV Loop ---
print(f"4. Starting {N_SPLITS}-Fold Cross-Validation...")
for train_index, test_index in skf.split(X_hog_full, y_full):
    print(f"   -> Training Fold {fold_count}/{N_SPLITS}...")
    
    X_train, X_test = X_hog_full[train_index], X_hog_full[test_index]
    y_train, y_test = y_full[train_index], y_full[test_index]
    
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    
    print(f"      Fold Accuracy: {accuracy_scores[-1]:.4f}")
    fold_count += 1

# --- 5. Final Detailed Results ---
print("\n" + "="*80)
print("✅ CROSS-VALIDATION ANALYSIS SUMMARY")
print("="*80)

mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

print(f"Validation Method: Stratified {N_SPLITS}-Fold CV")
print(f"Mean Fold Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f} SD)")

# Detailed Classification Report
az_labels = get_az_labels()
print("\nDETAILED CLASSIFICATION REPORT (Precision/Recall/F1-Score per Character):\n")
print(classification_report(y_true_all, y_pred_all, target_names=az_labels))


# --- 6. Visualization: Confusion Matrix (Raw Counts, A-Z, Gold) ---
print("\nGenerating Confusion Matrix (Raw Counts)...")
cm = confusion_matrix(y_true_all, y_pred_all)

plt.figure(figsize=(20, 18))
sns.heatmap(cm, 
            annot=True,     # Show raw counts (quantities)
            fmt="d",        # Format as integer
            cmap="YlOrBr",  # Gold-like color map
            cbar=True,
            xticklabels=az_labels, 
            yticklabels=az_labels,
            linewidths=.5,
            linecolor='black')

plt.title(f'Confusion Matrix (Raw Counts) for HOG+SVM Classification (K={N_SPLITS} Fold CV)\nMean Accuracy: {mean_accuracy:.4f}', fontsize=16)
plt.ylabel('True Label (A-Z)', fontsize=14)
plt.xlabel('Predicted Label (A-Z)', fontsize=14)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()