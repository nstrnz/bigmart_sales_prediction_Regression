# 3. Model Selection
## Importing the necessary libraries
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 4. Training and Evoluation
## 1-4 Data Spliting
# splitting the dataset into the Training set and Test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

## 2-4 use an appropriation Metric (MSE, RMSE, R2)
### Linear Regression
# create object for the LR 
lr_model = LinearRegression()

# fitting the model with training data 
lr_model = lr_model.fit(X_train, y_train)

# make predictions
y_pred_lr = lr_model.predict(X_test) 

# get result
mse_lr = mean_squared_error(y_test, y_pred_lr)
rscore_lr = r2_score(y_test, y_pred_lr)

print('MSE_LR: ', mse_lr)
print('RMSE_LR: ', np.sqrt(mse_lr))
print('R2_Score_LR: ', rscore_lr * 100)

### Decision Tree
# creating and fitting the model for DT
dt_model = DecisionTreeRegressor().fit(X_train, y_train)

# make prediction
y_pred_dt = dt_model.predict(X_test) 

# get result
mse_dt = mean_squared_error(y_test, y_pred_dt)
rscore_dt = r2_score(y_test, y_pred_dt)

print('MSE_DT: ', mse_dt)
print('RMSE_DT: ', np.sqrt(mse_dt))
print('R2_Score_DT: ', rscore_dt * 100)

### Random Forest
# creating and fitting the model for RF
rf_model = RandomForestRegressor().fit(X_train, y_train)

# make prediction
y_pred_rf = dt_model.predict(X_test) 

# get result
mse_rf = mean_squared_error(y_test, y_pred_rf)
rscore_rf = r2_score(y_test, y_pred_rf)

print('MSE_RF: ', mse_rf)
print('RMSE_RF: ', np.sqrt(mse_rf))
print('R2_Score_RF: ', rscore_rf * 100)