import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = pd.read_csv("irisFlower/IRIS.csv")

iris_copy = iris.copy()

le = LabelEncoder()
iris_copy['species'] = le.fit_transform(iris_copy['species'])
print(iris_copy['species'].value_counts())

x = iris_copy.drop('species', axis = 1)
y = iris_copy['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

#model training - RFC
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

# print("Random forest Accurancy", accuracy_score(y_test, y_pred_rf))
# print("Random Forect Classification Report: ", classification_report(y_test, y_pred_rf))

importances = rf.feature_importances_
feature_names = x.columns

sns.barplot(x=importances, y=feature_names, palette="coolwarm")
plt.title("Feature Importances")
plt.show()