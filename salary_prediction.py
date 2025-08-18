import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# loading the data
df = pd.read_csv("Salary_dataset.csv", encoding='utf-8', delimiter=',')
print(df.head())



# cleaning the data - checking missing or duplicate values in the dataset
missing_values = df.isnull().sum()
duplicate_values = df.duplicated().sum()
print('missing values\n', missing_values)
print('\nduplicate values\n', duplicate_values)

# dropping the unnamed column as it is not neccessary
data = df.drop(columns=['Unnamed: 0'], errors='ignore')
print(data.head())

# rearranging the "yearsexperience" column in the ascending order as the years column is not arranged
data = data.sort_values(by='YearsExperience', ascending=True)
print(data)

# using graphs for better understanding how the data is distributed
x = data['YearsExperience']
y = data['Salary']
plt.plot(x, y, color = 'red', marker='o', label ='data points')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Years of Experience vs Salary')
plt.tight_layout()
plt.savefig("visualization.png")
plt.show()

# split into train-test
X = data[['YearsExperience']]
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# trainning the model by linear regression model 
model = LinearRegression()
model.fit(X_train, y_train)


# making the predictions
user_input = float(input('\nenter your years of experience : '))
user_pred = model.predict([[user_input]])
print("based on your years of experience your salary would be : ", round(user_pred[0], 2))

#  CALCULATING ERROR :

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rse = r2_score(y_test, y_pred)

print('mean absolute error:', mae)
print('mean squared error:', mse)
print('r squared error:', rse)




