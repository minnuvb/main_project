#loan approval prediction 
#import libraries to this
import pandas as pd
import numpy as np
import seaborn as sns
loan=pd.read_csv('loan_data.csv')
loan.head()
loan.shape
loan.info()
loan.dtypes
print(loan.isnull().sum())
loan.describe()
from scipy.stats import zscore

# Columns to check for outliers..
numerical_columns = [
    'person_income', 'person_emp_exp', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'credit_score'
]

# Z-Score Outlier Detection. for cleaning

print("Outliers detected using Z-Score:")
for column in numerical_columns:
    loan[f'{column}_zscore'] = zscore(loan[column])
    outliers_zscore = loan[loan[f'{column}_zscore'].abs() > 3]
    print(f"\nColumn: {column}")
    print("outlier\n",outliers_zscore)
# Remove rows with outliers based on Z-Score
def remove_outliers_zscore(data, columns, threshold=3):
    """
    Removes rows with Z-Score outliers from the specified columns.
    Parameters:
        data: DataFrame
        columns: List of column names to check for outliers
        threshold: Z-score threshold (default is 3)
    Returns:
        DataFrame with outliers removed and without Z-score columns
    """
    for column in columns:
        data = data[data[f'{column}_zscore'].abs() <= threshold]
    
    # Drop Z-score columns
    zscore_columns = [f'{col}_zscore' for col in columns]
    data = data.drop(columns=zscore_columns, errors='ignore')
    
    return data

# Columns to check for outliers
numerical_columns = [
    'person_income', 'person_emp_exp', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'credit_score'
]

# Remove outliers
loan_cleaned = remove_outliers_zscore(loan, numerical_columns)

# Check the shape before and after removing outliers
print("shape of Dataset after finding z_score:", loan.shape)
print("Dataset Shape After Removing Outliers:", loan_cleaned.shape)

# Save the cleaned dataset without Z-score columns
loan_cleaned.to_csv("loan_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'loan_cleaned.csv'.")


# Define the ordinal encoding mappings..
ordinal_mapping = {
    "person_gender": {"male": 1, "female": 0},
    "person_education": {"High School": 0, "Bachelor": 1, "Master": 2, "Doctorate": 3},
    "person_home_ownership": {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3},
    "loan_intent": {
        "PERSONAL": 0,
        "EDUCATION": 1,
        "MEDICAL": 2,
        "VENTURE": 3,
        "HOMEIMPROVEMENT": 4,
        "DEBTCONSOLIDATION": 5,
    },
    "previous_loan_defaults_on_file": {"No": 0, "Yes": 1},
}

# Create a copy of the dataset to modify
loan_data_ordinal = loan_data_cleaned.copy()

for column, mapping in ordinal_mapping.items():
    # Map each value to its ordinal encoding
    loan_data_ordinal[column] = loan_data_cleaned[column].map(mapping)

# Display the first few rows of the updated dataset
loan_data_ordinal.head()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
sns.histplot(loan_data_ordinal['credit_score'], kde=True, bins=30, color='blue')
plt.title('Distribution of Credit Score', fontsize=16)
plt.xlabel('Credit Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.figure(figsize=(8, 4))
sns.histplot(loan_data_ordinal['person_income'], kde=True, bins=30, color='blue')
plt.title('Distribution of person income', fontsize=16)
plt.xlabel('Person Income', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.figure(figsize=(8, 4))
sns.histplot(loan_data_ordinal['person_gender'], kde=True, bins=30, color='blue')
plt.title('Distribution of person gender', fontsize=16)
plt.xlabel('Person Gender', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
correlation_matrix = loan_data_ordinal.corr()

# Set up the matplotlib figure for analysis
plt.figure(figsize=(12, 8))

# Create a heatmap
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm", 
    linewidths=0.5, 
    cbar=True
)

# Add title to the heatmap
plt.title("Correlation Heatmap of Loan Data", fontsize=16)

# Show the plot
plt.show()
plt.figure(figsize=(8, 4))
sns.countplot(x=loan_data_ordinal['person_home_ownership'])
plt.title('Count of Person Home Ownership', fontsize=16)
plt.xlabel('Person Home Ownership', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.figure(figsize=(8, 4))
sns.countplot(x=loan_data_ordinal['person_age'])
plt.title('Count of Person age', fontsize=16)
plt.xlabel('Person age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.figure(figsize=(8, 4))
sns.countplot(x=loan_data_ordinal['person_education'])
plt.title('Count of Person education', fontsize=16)
plt.xlabel('Person Home education', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.figure(figsize=(8, 4))
sns.scatterplot(
    x=loan_data_ordinal['credit_score'], 
    y=loan_data_ordinal['loan_amnt'], 
    hue=loan_data_ordinal['loan_status'], 
    alpha=0.8
)
plt.title('Scatterplot of Loan Amount vs Credit Score', fontsize=16)
plt.xlabel('Credit Score', fontsize=12)
plt.ylabel('Loan Amount', fontsize=12)
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.legend(title='Loan Status', loc='upper left', fontsize=10)
plt.show()
from sklearn.model_selection import train_test_split

# Assuming 'loan_data_ordinal' is the dataset after encoding and cleaning

# Define the features (X) and target (y)
X = loan_data_ordinal.drop(columns=['loan_status'])  
y = loan_data_ordinal['loan_status'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting datasets
print("Training set features shape:", X_train.shape)
print("Testing set features shape:", X_test.shape)
from imblearn.over_sampling import SMOTENC
import numpy as np

# Define categorical features (use column indices or column names)
categorical_features = [
    'person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file'
]

categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

# Initialize SMOTENC
smotenc = SMOTENC(categorical_features=categorical_indices, random_state=42)

# Apply SMOTENC on the training data
X_train_res, y_train_res = smotenc.fit_resample(X_train, y_train)

# Display the shape of the resampled dataset
print("Original training set shape:", X_train.shape)
print("Resampled training set shape:", X_train_res.shape)
print("Resampled Training Features (X_train_res):")
X_train_res.head()
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_res_scaled = scaler.fit_transform(X_train_res_imputed)


X_test_scaled = scaler.transform(X_test_imputed)

# train the model
logreg = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
logreg.fit(X_train_res_scaled, y_train_res)

# Make predictions
y_pred = logreg.predict(X_test_scaled)

# key evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test_scaled)[:, 1])
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
# Plot ROC Curve for checkinh how well model seperate approval and rejections
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
#next model decision tree






