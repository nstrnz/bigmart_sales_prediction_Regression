##Linear Regression

#  Predict
y_pred = model.predict(X_test)

# Calculate regression metrics
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error
print("r2_score:",r2_score(y_test,y_pred))
print("MAE:", mean_absolute_error(y_test,y_pred))
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))

##Random Forest
from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor()
rf.fit(X_train,y_train)

#Predict
Y_pred_rf= rf.predict(X_test)

# Calculate regression metrics
print("r2_score:",r2_score(y_test,Y_pred_rf))
print("MAE:",mean_absolute_error(y_test,Y_pred_rf))
print("MSE:",np.sqrt(mean_squared_error(y_test,Y_pred_rf)))



