import pandas as pd
import numpy as np

# Load the dataset
dataset_file = 'CEAS_08.csv'
df = pd.read_csv(dataset_file)

# Apply heuristics to label phishing emails
def heuristic_label(subject, body):
    phishing_keywords = ['prize', 'win', 'urgent', 'click here', 'free', 'account', 'verify']
    phishing_score = 0

    # Check for keywords in the subject
    if isinstance(subject, str):
        for keyword in phishing_keywords:
            if keyword in subject.lower():
                phishing_score += 1

    # Check for keywords in the body
    if isinstance(body, str):
        for keyword in phishing_keywords:
            if keyword in body.lower():
                phishing_score += 1

    # If phishing_score is above a certain threshold, classify as phishing (1)
    return 1 if phishing_score >= 2 else 0

# Apply the heuristic_label function to each row
df['label'] = df.apply(lambda x: heuristic_label(x['subject'], x['body']), axis=1)

# Save the updated dataset with labels
df.to_csv('CEAS_08_labeled.csv', index=False)
print("Updated dataset saved as 'CEAS_08_labeled.csv'")
