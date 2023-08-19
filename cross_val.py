
##LinearRegression

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# Create a cross-validation object with desired number of folds
num_folds = 5
cross_validation = KFold(n_splits=num_folds, shuffle=True)

# Perform cross-validation
cross_val_scores = cross_val_score(model, X, y, cv=cross_validation, scoring='neg_mean_squared_error')

# Note: The 'scoring' parameter can be adjusted to 'r2' for R-squared measurement or other relevant metrics

# Print the cross-validation scores
print("Cross-Validation Scores: ", cross_val_scores)

# Calculate and print the mean and standard deviation of the cross-validation scores
print("Mean of Cross-Validation Scores: ", cross_val_scores.mean())
print("Standard Deviation of Cross-Validation Scores: ", cross_val_scores.std())


##Random Forest

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# Create a cross-validation object with desired number of folds
num_folds = 5
cross_validation = KFold(n_splits=num_folds, shuffle=True)

# Perform cross-validation
cross_val_scores = cross_val_score(rf, X, y, cv=cross_validation, scoring='neg_mean_squared_error')

# Note: The 'scoring' parameter can be adjusted to 'r2' for R-squared measurement or other relevant metrics

# Print the cross-validation scores
print("Cross-Validation Scores: ", cross_val_scores)

# Calculate and print the mean and standard deviation of the cross-validation scores
print("Mean of Cross-Validation Scores: ", cross_val_scores.mean())
print("Standard Deviation of Cross-Validation Scores: ", cross_val_scores.std())




##Decision Tree

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



# Create a cross-validation object with desired number of folds
num_folds = 5
cross_validation = KFold(n_splits=num_folds, shuffle=True)

# Perform cross-validation
cross_val_scores = cross_val_score(DT, X, y, cv=cross_validation, scoring='neg_mean_squared_error')

# Note: The 'scoring' parameter can be adjusted to 'r2' for R-squared measurement or other relevant metrics

# Print the cross-validation scores
print("Cross-Validation Scores: ", cross_val_scores)

# Calculate and print the mean and standard deviation of the cross-validation scores
print("Mean of Cross-Validation Scores: ", cross_val_scores.mean())
print("Standard Deviation of Cross-Validation Scores: ", cross_val_scores.std())
