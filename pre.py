# 1. Data Preprocessing
## Importing the Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

## Importing the Dataset
dataset = pd.read_csv('bigmart.csv')

# datatype of attributes
dataset.info()

dataset.describe()

# replace reapted values in Item_Fat_Content
dataset['Item_Fat_Content'] = dataset.Item_Fat_Content.replace(['LF', 'low fat', 'reg'], ['Low Fat','Low Fat', 'Regular'])
dataset.Item_Fat_Content.value_counts()

# removing the ineffective columns
dataset = dataset.drop(columns = ['Item_Identifier' , 'Outlet_Identifier'], axis = 1)

## 1-1 Missing Values
dataset.isnull().sum()

# filling the object values with mode and float type with mean
dataset['Item_Weight'].fillna(dataset['Item_Weight'].mean(), inplace = True)

dataset['Outlet_Size'].fillna(dataset['Outlet_Size'].mode()[0], inplace = True)

## 2-1 Outliers
# detecting for outliers
plt.figure(figsize = (12, 6), dpi = 480)

plt.subplot(2,3,1)
sns.boxplot(dataset['Item_Weight'])

plt.subplot(2,3,2)
sns.boxplot(dataset['Item_Visibility'])

plt.subplot(2,3,3)
sns.boxplot(dataset['Item_MRP'])

plt.subplot(2,3,4)
sns.boxplot(dataset['Outlet_Establishment_Year'])

plt.subplot(2,3,5)
sns.boxplot(dataset['Item_Outlet_Sales'])

# removing ouliers
def outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    return upper_limit, lower_limit

upper_v, lower_v = outliers(dataset, "Item_Visibility")
upper_o, lower_o = outliers(dataset, "Item_Outlet_Sales")

dataset_vis = dataset[(dataset['Item_Visibility'] > lower_v) & (dataset['Item_Visibility'] < upper_v)]
dataset_sal = dataset[(dataset['Item_Outlet_Sales'] > lower_o) & (dataset['Item_Outlet_Sales'] < upper_o)]

plt.figure(figsize = (12, 6), dpi = 480)

plt.subplot(1,2,1)
sns.boxplot(dataset_vis['Item_Visibility'])
plt.title('Item_Visibility Distribution after removing outliers')

plt.subplot(1,2,2)
sns.boxplot(dataset_sal['Item_Outlet_Sales'])
plt.title('Item Outlet Sales Distribution after removing outliers')

## 3-1 Encoging categorical data
data = pd.get_dummies(dataset, columns = ['Item_Type'])

le = LabelEncoder()

cat_col = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

for col in cat_col:
    data[col] = le.fit_transform(dataset[col])

## 4-1 Normalizing
# Data inputs
X = data.drop(['Item_Outlet_Sales'], axis = 1)
copy_x = X

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)
X = pd.DataFrame(X)
X.columns = copy_x.columns

# Data outputs
y = data['Item_Outlet_Sales']

# 2. Feature Selection
## 1-2 Correlation Analysis
plt.figure(figsize = (12, 8), dpi = 480)
sns.heatmap(dataset.corr(), annot = True, cmap = "crest", linewidths = 0.5, linecolor = 'black')
plt.show()

## 2-2 Features Importance
# create and fit the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# access feature importance insights(coefficients)
features_label = X.columns
importances = pd.Series(model.coef_, index = X.columns)
indices = np.argsort(importances)[::-1]
for i in range(X.shape[1]):
    print ("%2d) %-*s %f" % (i + 1, 30, features_label[i], importances[indices[i]]))
    
# visualize feature importance (optional)
plt.figure(figsize = (12, 6), dpi = 480)
plt.barh(importances.index, importances.values)
plt.xlabel('coefficient Values', fontsize = 12)
plt.ylabel('features', fontsize = 12)
plt.title('Features Importance', fontsize = 20)
plt.show()