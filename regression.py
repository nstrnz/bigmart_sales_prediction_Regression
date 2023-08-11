##Linear Regression

# Predict the target variable for the test data (LR)

y_pred = model.predict(X_test)

# Evaluate the model

from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error
print("r2_score:",r2_score(y_test,y_pred))
print("MAE:", mean_absolute_error(y_test,y_pred))
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))

##Random Forest

from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor()
rf.fit(X_train,y_train)

# Predict the target variable for the test data (RF)

Y_pred_rf= rf.predict(X_test)

# Evaluate the model

print("r2_score:",r2_score(y_test,Y_pred_rf))
print("MAE:",mean_absolute_error(y_test,Y_pred_rf))
print("MSE:",np.sqrt(mean_squared_error(y_test,Y_pred_rf)))

##Decision Tree

from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor(max_depth=5)
DT.fit(X_train, y_train)

# Predict the target variable for the test data (DT)

y_pred_DT = DT.predict(X_test)

# Evaluate the model

print("r2_score:",r2_score(y_test,y_pred_DT))
print("MAE:",mean_absolute_error(y_test,y_pred_DT))
print("MSE:",np.sqrt(mean_squared_error(y_test,y_pred_DT)


