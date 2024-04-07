import pandas as pd

# Load data
data = pd.read_excel("C:/Users/stdso/Documents/USTH/Med/BioAct-Het-main/Data/Results.xlsx")

# Calculating side effects > 0.5 for each drug
se_count = (data.set_index('Drug') > 0.5).sum(axis=1).sort_values(ascending=False)

# Count of drugs with each side effect > 0.5
common = data.set_index('Drug').gt(0.5).sum(axis=0).sort_values(ascending=False)

# Print results
print("Drugs with Side Effects from Most to Least:")
print(se_count.to_string(), "\n")

print("Common Side Effects Across All Drugs from Most to Least:")
print(common.to_string())

