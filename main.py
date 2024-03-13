import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

data = pd.read_csv("melb_data.csv")

# Selecting target
y = data.Price

# Numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Dividing data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=0)


# Define a function to measure the quality of different approaches
def score_dataset(X_train_local, X_test_local, y_train_local, y_test_local):
    """

    :param X_train_local: Train subset of feature data
    :param X_test_local:  Test subset of feature data
    :param y_train_local: Train subset of target data
    :param y_test_local:  Test subset of target data
    :return: The function reports the mean absolute error or MAE from
    our random forest model
    """
    model = RandomForestRegressor(n_estimators=10,
                                  random_state=0)
    model.fit(X_train_local, y_train_local)
    predicted = model.predict(X_test_local)
    return mean_absolute_error(y_test_local, predicted)


# Getting the names of columns with missing values
cols_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# Dropping columns in training and testing data
reduced_X_train = X_train.drop(cols_missing, axis=1)
reduced_X_test = X_test.drop(cols_missing, axis=1)

print("Mae from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))


# Using SimpleImputer to replace missing values with the mean value along each column

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))

imputed_X_train.columns = X_train.columns
imputed_X_test.columns = X_test.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

# Approach 3 - Extension to Imputation
X_train_plus = X_train.copy()
X_test_plus = X_test.copy()

# New column indicating what will be imputed
for col in cols_missing:
    X_train_plus[col + "_was_missing"] = X_train_plus[col].isnull()
    X_test_plus[col + "_was_missing"] = X_test_plus[col].isnull()

my_imputer2 = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer2.fit_transform(X_train_plus))
imputed_X_test_plus = pd.DataFrame(my_imputer2.fit_transform(X_test_plus))

print("MAE for approach 3 (Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

