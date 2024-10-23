import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import seaborn as sns
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

# Step 1: Load the Labeled Dataset
dataset_file = 'CEAS_08_labeled.csv'  # Make sure you have the labeled dataset

try:
    df = pd.read_csv(dataset_file)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{dataset_file}' was not found. Please ensure it's in the correct directory.")
    exit()

# Step 2: Use 'body' as the text for phishing detection
text_column = 'body'
label_column = 'label'

print(f"Columns in dataset: {df.columns}")

# Optional: Reduce dataset size for faster testing
# df = df.sample(1000, random_state=42)

# Step 3: Clean the Text Data by Removing Punctuation and Stopwords
print("Starting text cleaning...")
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])  # Remove stopwords
    return text

# Apply the cleaning function to the email body column
df['body_clean'] = df[text_column].apply(clean_text)
print("Text cleaning completed.")

# Step 4: Convert Text to Numerical Features Using TF-IDF
print("Converting text to numerical features using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['body_clean']).toarray()  # Features (numerical representation of emails)
y = df[label_column]  # Labels (1 for phishing, 0 for legitimate)
print("TF-IDF conversion completed.")

# Step 5: Balance the Dataset Using SMOTE
print("Balancing the dataset using SMOTE...")
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
print(f"Dataset balanced. New class distribution: \n{pd.Series(y).value_counts()}")

# Step 6: Split the Dataset into Training and Testing Sets
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Dataset splitting completed.")

# Step 7: Train and Evaluate Logistic Regression Model
print("Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=500, class_weight='balanced')
lr_model.fit(X_train, y_train)
print("Logistic Regression model training completed.")

# Evaluate Logistic Regression Model
print("Evaluating the Logistic Regression model...")
y_pred_lr = lr_model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f"\nLogistic Regression - Accuracy: {accuracy_lr:.4f}")
print(f"Logistic Regression - Precision: {precision_lr:.4f}")
print(f"Logistic Regression - Recall: {recall_lr:.4f}")
print(f"Logistic Regression - F1-Score: {f1_lr:.4f}")

# Step 8: Train and Evaluate Random Forest Model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
print("Random Forest model training completed.")

# Evaluate Random Forest Model
print("Evaluating the Random Forest model...")
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"\nRandom Forest - Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest - Precision: {precision_rf:.4f}")
print(f"Random Forest - Recall: {recall_rf:.4f}")
print(f"Random Forest - F1-Score: {f1_rf:.4f}")

# Step 9: Train and Evaluate SVM Model
print("Training SVM model...")
svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_model.fit(X_train, y_train)
print("SVM model training completed.")

# Evaluate SVM Model
print("Evaluating the SVM model...")
y_pred_svm = svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

print(f"\nSVM - Accuracy: {accuracy_svm:.4f}")
print(f"SVM - Precision: {precision_svm:.4f}")
print(f"SVM - Recall: {recall_svm:.4f}")
print(f"SVM - F1-Score: {f1_svm:.4f}")

# Step 10: Plot Confusion Matrices for All Models
print("Plotting confusion matrices...")
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(18, 5))

# Confusion Matrix for Logistic Regression
plt.subplot(1, 3, 1)
sns.heatmap(conf_matrix_lr, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression')

# Confusion Matrix for Random Forest
plt.subplot(1, 3, 2)
sns.heatmap(conf_matrix_rf, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest')

# Confusion Matrix for SVM
plt.subplot(1, 3, 3)
sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for SVM')

plt.tight_layout()
plt.show()

# Step 11: Display Emails and Their Predicted Labels for All Models
print("\nDisplaying the predictions for the test set using Logistic Regression, Random Forest, and SVM...")

# Reverse the TF-IDF transformation for better interpretability
X_test_original = tfidf.inverse_transform(X_test)

# Create DataFrames to display the actual email, the true label, and the predicted labels
predictions_df = pd.DataFrame({
    'Email Body': [" ".join(tokens) for tokens in X_test_original],  # Reconstructed email text
    'True Label': y_test,
    'Predicted Label (LR)': y_pred_lr,
    'Predicted Label (RF)': y_pred_rf,
    'Predicted Label (SVM)': y_pred_svm
})

# Display the first 10 predictions
print(predictions_df.head(10))

# Save the predictions to a CSV file (optional)
predictions_df.to_csv('predictions_output_all_models.csv', index=False)
print("\nPredictions for all models saved to 'predictions_output_all_models.csv'.")
