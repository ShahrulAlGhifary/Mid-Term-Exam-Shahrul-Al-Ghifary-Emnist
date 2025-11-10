import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import string
import os
import sys
from skimage.feature import hog
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# --- File Path ---
FILE_PATH = "D:\\UTS Machine Vision\\emnist-letters-train.csv"

# --- Data Sampling Parameters (from Code A) ---
NUM_CLASSES = 26
SAMPLES_PER_CLASS = 500
TOTAL_SAMPLES = NUM_CLASSES * SAMPLES_PER_CLASS

# --- Optimal Model Parameters (from Code B) ---
OPTIMAL_C = 10
OPTIMAL_GAMMA = 'scale'
KERNEL = 'rbf'

# --- Helper Function (from Code A) ---
def fix_orientation(img):
    """Fixes the EMNIST image orientation from the CSV format."""
    img_fixed = np.fliplr(img.T)
    return img_fixed

# --- Helper Function (from Code B) ---
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
    
    # HOG Parameters (fixed from Code B)
    features = hog(resized_image, 
                   orientations=9, 
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), 
                   visualize=False, 
                   transform_sqrt=True,
                   feature_vector=True)
    return features

# --- Execution Starts Here ---

print("--- Step 1 : Loading Dataset ---")
try:
    print(f"Reading CSV file from : {FILE_PATH}")
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"File not found at: {FILE_PATH}")
        
    data_frame = pd.read_csv(FILE_PATH, header=None)
    print("CSV file loaded successfully.")
    print(f"Total data loaded : {len(data_frame)} samples")
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

labels_full = data_frame.iloc[:, 0].values
images_flat = data_frame.iloc[:, 1:].values.astype('uint8')

# Apply Code A's orientation fix
images_raw = images_flat.reshape(-1, 28, 28)
print("Fixing image orientation (rotate & flip)...")
images_full = np.array([fix_orientation(img) for img in images_raw])
print("Image orientation fixed.\n")


print("--- Step 2 : Performing Data Sampling (Code A's Method) ---")
sampled_images = []
sampled_labels = []

print(f"Taking {SAMPLES_PER_CLASS} samples per class for {NUM_CLASSES} classes...")
for i in range(1, NUM_CLASSES + 1):
    class_indices = np.where(labels_full == i)[0]
    # Handle cases where a class might have fewer than SAMPLES_PER_CLASS (unlikely for EMNIST train)
    if len(class_indices) < SAMPLES_PER_CLASS:
        print(f"Warning: Class {i} has only {len(class_indices)} samples. Using all of them.")
        random_indices = class_indices
    else:
        random_indices = np.random.choice(class_indices, SAMPLES_PER_CLASS, replace=False)
    
    sampled_images.append(images_full[random_indices])
    sampled_labels.append(labels_full[random_indices])

X_data = np.concatenate(sampled_images, axis=0)
y_data = np.concatenate(sampled_labels, axis=0)
print(f"Sampling complete. Total data for this experiment : {X_data.shape[0]} samples.\n")


print("--- Step 3 : HOG Feature Extraction ---")
hog_features = []
start_time = time.time()

print("Processing images for HOG feature extraction...")
# Use tqdm progress bar (from Code B) on the sampled data (from Code A)
for image in tqdm(X_data, desc="HOG Extraction"):
    features = extract_hog_features(image) # Use Code B's HOG function
    hog_features.append(features)

X_features = np.array(hog_features)
end_time = time.time()
print(f"HOG extraction finished in {end_time - start_time:.2f} seconds.")
print(f"HOG feature dataset shape : {X_features.shape}\n")


print("--- Step 4 : Model Evaluation with LOOCV (Code A's Method) ---")
# Use optimal parameters from Code B
model = SVC(kernel=KERNEL, C=OPTIMAL_C, gamma=OPTIMAL_GAMMA, random_state=42)
print(f"SVM model prepared with kernel ='{KERNEL}', C={OPTIMAL_C}, and gamma={OPTIMAL_GAMMA}")

# Use LOOCV from Code A
loo = LeaveOneOut()
print(f"Validation method: Leave-One-Out (will run {X_data.shape[0]} iterations).\n")

print("===============================================================")
print("STARTING LOOCV EVALUATION.....)")
print("The process will take a long time, do not close the application!!!.....")
print("===============================================================")
start_cv_time = time.time()

y_pred = cross_val_predict(model, X_features, y_data, cv=loo, n_jobs=-1)

end_cv_time = time.time()
total_minutes = (end_cv_time - start_cv_time) / 60
print(f"\nLOOCV evaluation finished in {total_minutes:.2f} minutes.\n")


print("--- Step 5 : Displaying Performance Results (Code A's Style) ---")

# Calculate Accuracy
accuracy = accuracy_score(y_data, y_pred)
print(f"Accuracy: {accuracy * 100:.4f}%\n")

# Display Classification Report (Precision, Recall, F1-Score)
print("Classification Report (Precision, Recall, F1-Score):")
report_labels = list(range(1, NUM_CLASSES + 1))
target_names = [chr(ord('A') + i - 1) for i in report_labels]
report = classification_report(y_data, y_pred, labels=report_labels, target_names=target_names, digits=4)
print(report)

# Create and Display Confusion Matrix (Code A's Style)
print("Generating Confusion Matrix plot...")
cm = confusion_matrix(y_data, y_pred, labels=report_labels)

plt.figure(figsize=(18, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrBr', xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix (LOOCV - HOG + SVM {KERNEL.capitalize()})', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()

try:
    plt.savefig('confusion_matrix_loocv.png', dpi=300)
    print("Confusion Matrix plot saved as 'confusion_matrix_loocv.png'")
except Exception as e:
    print(f"Failed to save plot: {e}")

plt.show()

# --- Final Printout (from Code A) ---
print("\n=================================================") 
print("Name     : Shahrul Al Ghifary ")
print("NIM      : 4212301012")
print("Class    : Mechatronics 5A - Morning")
print("=================================================")