import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dts
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic = pd.read_csv("titanic/Titanic-Dataset.csv")


#Dropping columns and cleaning data
titanic = titanic.drop(['Name', 'Ticket', 'Cabin'], axis=1)

titanic['Sex'] = titanic['Sex'].map({'male' : 0, 'female' : 1})
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])
titanic = pd.get_dummies(titanic, columns = ['Embarked'], drop_first = True)

#splitting data
x = titanic.drop('Survived', axis=1)
y = titanic['Survived']

#model
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

survivors = titanic['Survived'].value_counts()

plt.figure(figsize=(20, 10))
survivors.plot(kind='bar', color=['black', 'Blue'])
plt.title("Survival count on the Titanic Incident")
plt.xlabel("Survivors and Deceased")
plt.ylabel("Number of Survivors")
plt.xticks([0, 1], ['Deceased', 'Survived'], rotation=0)  
plt.show()

