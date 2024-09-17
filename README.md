# Bioactivity Prediction by a Heterogeneous Siamese Neural Network (SNN) 

## Group 2 Machine Learning in Medicine - Year 3 Data Science
- Pham Hai Nam (BI12-307)
- Nguyen Son (BI12-389)
- Le Mau Minh Phuc (BI12-351)
- Doan Huu Thanh (BI12-418) 
*University of Science and Technology of Hanoi*

## Abstract
The determination of bioactivity during medication leads to challenges that can cause drug failure. This study presents a model, BioAct-Het, which utilizes a Siamese Neural Network (SNN) to predict bioactivity classes for both COVID-19 and non-COVID-19 drugs. The model demonstrates an accuracy greater than 75%, making it a reliable tool for evaluating side effects and enhancing drug discovery.

## Introduction
### Context of Computational Methods in Drug Discovery
The process of drug discovery is lengthy and costly, often taking over a decade and requiring billions of dollars in investment. Predicting a compound's bioactivities—both positive effects and potential side effects—is crucial for successful drug development. Computational methods, particularly those that explore structure-activity relationships (SAR), provide an efficient alternative to traditional laboratory screening.

### Objectives
1. **Evaluate the BioAct-Het model** using predicted and actual side effect data from a set of known drugs.
2. **Identify bioactivity classes** for 24 marketed drugs, including both COVID-19 and non-COVID-19 treatments.

## Methodology
![image](https://github.com/user-attachments/assets/1980fe67-9c51-43d0-9177-33213b41cb18)

### Data Materials
Four distinct datasets were utilized in this study:

1. **Manual Collection Dataset**: This dataset consists of 24 marketed drugs specifically used for treating COVID-19. The drugs were selected based on their documented bioactivities and side effects.

2. **SIDER (Side Effect Resource)**: SIDER provides information about the side effects of 1,427 marketed drugs, categorized into 27 classes based on the Medical Dictionary for Regulatory Activities (MedDRA). This dataset contains over 21,000 samples of side effects.

3. **Tox21**: This dataset includes 7,831 chemical substances evaluated through 12 toxicological assays. It is designed to assess potential risks associated with chemical exposures, aiding in the prediction of toxicity.

4. **MUV (Maximum Unbiased Validation)**: MUV serves as a benchmark for evaluating virtual screening methods. It includes over 93,000 compounds and is structured to minimize the risk of overfitting by presenting 17 challenging tasks.

### Model Architecture
The BioAct-Het model is built on a heterogeneous Siamese Neural Network architecture, which consists of two branches:
- **Chemical Representation Branch**: Utilizes Graph Convolutional Networks (GCN) to transform SMILES representations of chemical structures into vector embeddings.
- **Bioactivity Class Representation Branch**: Employs a binary string encoding (Morgan fingerprint) to represent bioactivity classes based on known active and inactive substructures.

### Evaluation Metrics
To assess the model's performance, several evaluation metrics were employed:
- **Accuracy**: Measures the overall correctness of the model's predictions.
- **Precision**: Evaluates the ratio of true positive predictions to the total predicted positives.
- **Recall**: Measures the ratio of true positive predictions to the total actual positives.
- **F1 Score**: Provides a balance between precision and recall, especially useful for imbalanced datasets.

## Results
### Model Evaluation
The BioAct-Het model was evaluated using a test set of seven drugs, comparing the predicted side effects with actual documented side effects from the SIDER dataset. The drugs included:
- Chloroquine
- Famotidine
- Guanfacine
- Hydroxychloroquine
- Oseltamivir
- Prednisolone
- Ritonavir

The model achieved an overall accuracy of **87.76%**, with individual metrics as follows:
![image](https://github.com/user-attachments/assets/28a6fbd0-2649-4ec0-a715-942ead91c34e)


### Drug Discovery Analysis
From the analysis, five drugs were identified with the fewest predicted side effects:
1. **Paxlovid (Nirmatrelvir)**: 12 predicted side effects.
2. **Linagliptin**: 11 predicted side effects.
3. **Interferon-beta-1a**: 11 predicted side effects.
4. **Famotidine**: 10 predicted side effects.
5. **Hydroxychloroquine**: 10 predicted side effects.
![image](https://github.com/user-attachments/assets/41ad0e7f-01a9-4ca2-95b9-a10a00baa5de)

Notably, Paxlovid and Interferon-beta-1a are both relevant to COVID-19 treatment, suggesting their potential use in clinical settings. Conversely, the model also identified drugs with the highest predicted side effects, indicating the need for careful monitoring:
- **Ritonavir**: 25 predicted side effects.
- **Ivermectin-B1a**: 26 predicted side effects.
- **Cyclosporine**: 26 predicted side effects.
![image](https://github.com/user-attachments/assets/1c000ebf-d80d-4161-8a50-1c1c47d4a780)

### Visualization
Confusion matrices and bubble charts were generated to visualize the model's prediction performance across the tested drugs. The confusion matrices indicated that the model accurately predicted both the presence and absence of side effects for most drug-side effect pairs, highlighting the model's reliability in clinical applications.

They can be found here for the [Bubble Charts](https://github.com/Xernnn/Bioact-Pred-Het-SNN/tree/main/Output/Bubble) and the [Confusion Matrices](https://github.com/Xernnn/Bioact-Pred-Het-SNN/tree/main/Output/CF)

## Conclusion
This study successfully demonstrates the application of the BioAct-Het model in predicting bioactivity classes and side effects of marketed drugs. The findings underscore the model's potential utility in drug discovery, particularly in identifying safe options for COVID-19 treatment. Further research is recommended to validate these findings through in vitro and in vivo studies.
