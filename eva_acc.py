import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the datasets
sidereva_df = pd.read_csv('C:/Users/stdso/Documents/USTH/Med/BioAct-Het-main/Data/sidereva.csv')
group2eva_df = pd.read_csv('C:/Users/stdso/Documents/USTH/Med/BioAct-Het-main/Data/group2eva.csv')

# Ensure 'Drug_Name' is of type string in both DataFrames
sidereva_df['Drug_Name'] = sidereva_df['Drug_Name'].astype(str)
group2eva_df['Drug_Name'] = group2eva_df['Drug_Name'].astype(str)

# Merge the dataframes on 'Drug_Name'
merged_df = pd.merge(sidereva_df, group2eva_df, on='Drug_Name', suffixes=('_gt', '_pred'))

# Define the ground truth and prediction columns
gt_columns = [col for col in merged_df.columns if col.endswith('_gt') and col not in ["0_gt", "smiles_gt"]]
pred_columns = [col.replace('_gt', '_pred') for col in gt_columns]

# Initialize a dictionary to store performance metrics for each drug
performance_metrics = {}

# Variables to calculate overall metrics
total_correct_predictions = 0
total_predictions = 0
y_true_all = []
y_pred_all = []

# Compute metrics for each drug
for drug_name in merged_df['Drug_Name'].unique():
    drug_df = merged_df[merged_df['Drug_Name'] == drug_name]
    y_true = drug_df[gt_columns].iloc[0].values.astype(int)
    y_pred = (drug_df[pred_columns].iloc[0].values >= 0.5).astype(int)
    
    # Concatenate all true labels and predictions for overall metrics calculation
    y_true_all.extend(y_true)
    y_pred_all.extend(y_pred)
    
    # Update overall accuracy calculation
    total_correct_predictions += (y_true == y_pred).sum()
    total_predictions += len(y_true)
    
    # Calculate metrics with 'weighted' average for multiclass targets
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Store metrics
    performance_metrics[drug_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Calculate overall metrics
overall_accuracy = total_correct_predictions / total_predictions
overall_precision = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
overall_recall = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
overall_f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)

# Convert the overall metrics into a DataFrame
overall_metrics_df = pd.DataFrame([{
    'Overall Accuracy': overall_accuracy,
    'Overall Precision': overall_precision,
    'Overall Recall': overall_recall,
    'Overall F1 Score': overall_f1
}])

# After computing the metrics for each drug:
metrics_list = []

# Convert the performance_metrics dictionary to a list of dictionaries for the DataFrame
for drug_name, metrics in performance_metrics.items():
    metrics['Drug_Name'] = drug_name  # Add the drug name to each dictionary
    metrics_list.append(metrics)

# Convert the list of dictionaries into a DataFrame
metrics_df = pd.DataFrame(metrics_list)

# Reorder columns so Drug_Name is first
columns = ['Drug_Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_df = metrics_df[columns]

# Display the DataFrames
print("Performance Metrics for Each Drug:")
print(metrics_df.to_string(index=False))
print("\nOverall Metrics:")
print(overall_metrics_df.to_string(index=False))
