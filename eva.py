import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from adjustText import adjust_text
import os

output_directory = 'C:/Users/stdso/Documents/USTH/Med/BioAct-Het-main/Output/Bubble'  # Replace with your actual path
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to calculate adjusted size for visibility
def scale_size(probability):
    return (probability**2) * 1800 + 200 

# Load the datasets
sidereva_path = 'C:/Users/stdso/Documents/USTH/Med/BioAct-Het-main/Data/sidereva.csv'
group2eva_path = 'C:/Users/stdso/Documents/USTH/Med/BioAct-Het-main/Data/group2eva.csv'

sidereva_df = pd.read_csv(sidereva_path)
group2eva_df = pd.read_csv(group2eva_path)

# Ensure 'Drug_Name' is of type string
sidereva_df['Drug_Name'] = sidereva_df['Drug_Name'].astype(str)
group2eva_df['Drug_Name'] = group2eva_df['Drug_Name'].astype(str)

# Merge the dataframes on 'Drug_Name'
merged_df = pd.merge(sidereva_df, group2eva_df, on='Drug_Name', suffixes=('_gt', '_pred'))

# Define category colors
category_colors = {
    'True Positive': 'blue',
    'True Negative': 'green',
    'False Positive': 'purple',
    'False Negative': 'red'
}

def adjust_text_position(ax, text_objects, x_values):
    from adjustText import adjust_text
    adjust_text(text_objects, x=x_values, arrowprops=dict(arrowstyle='-', color='black'))

# Iterate through each drug to generate bubble charts
for drug_name in merged_df['Drug_Name'].unique():
    drug_df = merged_df[merged_df['Drug_Name'] == drug_name]
    plotting_data = []
    
    disorder_columns = [col for col in sidereva_df.columns if col not in ['Unnamed: 0', 'smiles', 'Drug_Name']]
    for disorder in disorder_columns:
        disorder_gt = disorder + '_gt'
        disorder_pred = disorder + '_pred'
        ground_truth = drug_df[disorder_gt].iloc[0]
        probability = drug_df[disorder_pred].iloc[0]
        plotting_data.append({
            'Disorder': disorder,
            'Ground Truth': ground_truth,
            'Probability': probability
        })
    
    # Convert plotting data into DataFrame
    drug_plotting_df = pd.DataFrame(plotting_data)
    drug_plotting_df['Category'] = drug_plotting_df.apply(lambda row: 'True Positive' if row['Ground Truth'] == 1 and row['Probability'] >= 0.5
                                                          else 'False Negative' if row['Ground Truth'] == 1 and row['Probability'] < 0.5
                                                          else 'False Positive' if row['Ground Truth'] == 0 and row['Probability'] >= 0.5
                                                          else 'True Negative', axis=1)
    drug_plotting_df['Size'] = scale_size(drug_plotting_df['Probability'])
    drug_plotting_df['Color'] = drug_plotting_df['Category'].map(category_colors)

    # Plot the bubble chart for the drug
    fig, ax = plt.subplots(figsize=(10, 8))
    text_objects = []
    for index, row in drug_plotting_df.iterrows():
        ax.scatter(row['Ground Truth'], row['Probability'], 
                s=row['Size'], alpha=0.5, 
                color=row['Color'], marker='o')
        # Align text differently based on the Ground Truth value
        if row['Ground Truth'] == 0:
            ha = 'left'
            x_position = row['Ground Truth'] + 0.01
        else:
            ha = 'right'
            x_position = row['Ground Truth'] - 0.01
        text_obj = ax.text(x_position, row['Probability'], row['Disorder'],
                        ha=ha, va='center', fontsize=8)
        text_objects.append(text_obj)

    # Use adjust_text to prevent overlap
    adjust_text(text_objects, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.2)
    ax.set_xticks([0, 1])
    ax.set_title(f'{drug_name}')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Probability')
    ax.grid(True)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=category,
                            markerfacecolor=color, markersize=10) for category, color in category_colors.items()]
    ax.legend(handles=legend_elements, loc='upper center')

    plt.tight_layout()
    output_path = os.path.join(output_directory, f'{drug_name}_bubble_chart.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
