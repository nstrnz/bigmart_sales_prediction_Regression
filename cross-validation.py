# 5. Cross Validation
## Importing the necessary libraries
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

### k-fold cross validation
## Linear Regression
# define cross-validation method to use
kf_lr = KFold(n_splits = 5, random_state = 42, shuffle = True)

# perform cross-validation with 5 folds
scores_lr = cross_val_score(lr_model, X_train, y_train, cv = kf_lr, scoring = 'r2')

rscores_lr = np.mean(scores_lr)
rmsLR_kf = np.sqrt(np.mean(np.absolute(scores_lr)))

print('RMSE_kf: ', rmsLR_kf)
print('R2score_kf: ', rscores_lr)

## Decision Tree
# define cross-validation method to use
kf_dt = KFold(n_splits = 5, random_state = 42, shuffle = True)

# perform cross-validation with 5 folds
scores_dt = cross_val_score(dt_model, X_train, y_train, cv = kf_dt, scoring = 'r2')

rscores_dt = np.mean(scores_dt)
rmsDT_kf = np.sqrt(np.mean(np.absolute(scores_dt)))

print('RMSE_kf: ', rmsDT_kf)
print('R2score_kf: ', rscores_dt)

## Random Forest
# define cross-validation method to use
kf_rf = KFold(n_splits = 5, random_state = 42, shuffle = True)

# perform cross-validation with 5 folds
scores_rf = cross_val_score(rf_model, X_train, y_train, cv = kf_rf, scoring = 'r2')

rscores_rf = np.mean(scores_rf)
rmsRF_kf = np.sqrt(np.mean(np.absolute(scores_rf)))

print('RMSE_kf: ', rmsRF_kf)
print('R2score_kf: ', rscores_rf)

### GridSearchCV
## Linear Regression
# call and fit GridSearchCV
cv_lr = GridSearchCV(lr_model , param_grid = {'fit_intercept':[True, False]}, scoring = 'r2', cv = 5, verbose = 0)

# fit the LR model
cv_lr.fit(X_train, y_train)

# print best parameter and best score After tuning 
print("Best Parameters: ", cv_lr.best_params_)
print("\nBest Score: ", cv_lr.best_score_) 

# predict and print R2 score 
grid_pred_lr = cv_lr.predict(X_test)
r2score_lr = r2_score(y_test, grid_pred_lr)

print('\nR2score: ', r2score_lr)

## Decision Tree
# hyperparameters tuning with GridSearchCV
dt_params = {"min_samples_split": [10, 20, 40], "max_depth": [2, 6, 8], "min_samples_leaf": [20, 40, 100],
             "max_leaf_nodes": [5, 20, 100], 'min_samples_split': range(2, 10)}

# call GridSearchCV
cv_dt = GridSearchCV(dt_model, dt_params, scoring = 'r2', cv = 5, refit = True)

# fit the DT model
cv_dt.fit(X_train, y_train)

# print best parameter and best score After tuning 
print("Best Parameters: ", cv_dt.best_params_)
print("\nBest Score: ", cv_dt.best_score_) 

# predict and print R2 score 
grid_pred_dt = cv_dt.predict(X_test)
r2score_dt = r2_score(y_test, grid_pred_dt)

print('\nR2score: ', r2score_dt)

## Random Forest
# create the parameters grid
rf_params = {'n_estimators': [50, 100, 150, 200, 250], 'max_depth': [2, 4, 6], 'max_leaf_nodes': [5, 10],
            'max_features': ['auto', 'sqrt', 'log2'], 'min_samples_split': range(2, 10)}

# call GridSearchCV
cv_rf = GridSearchCV(rf_model, rf_params, scoring = 'r2', cv = 5, refit = True)

# fit the RF model
cv_rf.fit(X_train, y_train)

# print best parameter and best score After tuning 
print("Best Parameters: ", cv_rf.best_params_)
print("\nBest Score: ", cv_rf.best_score_) 

# predict and print R2 score 
grid_pred_rf = cv_rf.predict(X_test)
r2score_rf = r2_score(y_test, grid_pred_rf)

print('\nR2score: ', r2score_rf)

# comparing all the models
models = pd.DataFrame({'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
                       'R2-Score': [rscore_lr, rscore_dt, rscore_rf],
                       'R2-Score_KF': [rscores_lr, rscores_dt, rscores_rf],
                       'R2-Score_CV': [r2score_lr, r2score_dt, r2score_rf]})

models.sort_values(by = 'R2-Score', ascending = False)