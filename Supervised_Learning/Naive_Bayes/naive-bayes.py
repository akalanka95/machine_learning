import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error   

'''
We can use Scikit learn to build a simple linear regression model
you can use it use it like 
model = GaussianNB()

'''

#### Your Code Goes Here #### 
## Survived,Pclass,Name,Sex,Age,Siblings/Spouses Aboard,Parents/Children Aboard,Fare
## Step 1: Load Data from CSV File ####

dataframe = pd.read_csv("titanic.csv")

dataframe = dataframe.drop(["Name"],axis=1)

print dataframe.describe()

## Step 2: Plot the Data ####

ages = dataframe["Age"].values
fares = dataframe["Fare"].values
survived = dataframe["Survived"].values
colors = []
for item in survived:
	if item == 0:
		colors.append('red') 
	else:
		colors.append('green')	

#plt.scatter(ages,fares,s=50,color=colors)
#plt.show()

## Step 3: Build a NB Model ####

Features = dataframe.drop(['Survived'],axis=1).values
Targets = dataframe['Survived'].values

Features_train, Targets_train = Features[0:710], Targets[0:710]
Features_test, Targets_test = Features[710:], Targets[710:] 

model = GaussianNB()
model.fit(Features_train,Targets_train)
## Step 4: Print Predicted vs Actuals ####
predicted_values = model.predict(Features_test)

for item in zip(Targets_test,predicted_values):
	print "Actual was: ", item[0], "Predicted was: ", item[1]

## Step 5: Estimate Error ####
print "Accuracy is", model.score(Features_test,Targets_test) 