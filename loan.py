#predictive modeling for loan approval

from sklearn.preprocessing import OneHotEncoder

# Specify the categorical columns
categorical_columns = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]

# Initialize the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Fit and transform the categorical data
encoded_features = encoder.fit_transform(loan_data_cleaned[categorical_columns])

# Get the new column names for the encoded features
encoded_feature_names = encoder.get_feature_names_out(categorical_columns)

# Create a DataFrame for the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=loan_data_cleaned.index)

# Drop the original categorical columns from the dataset
loan_data_encoded = loan_data_cleaned.drop(columns=categorical_columns)

# Concatenate the encoded features with the original dataset
loan_data_encoded = pd.concat([loan_data_encoded, encoded_df], axis=1)

# Display the first few rows of the encoded dataset
loan_data_encoded.head()
