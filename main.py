import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

# Read in the data given
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


y = train_data["Survived"]

# Define the features
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# initiate the model
# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)
#
# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
# output.to_csv('final_answers.csv', index=False)
# print("Your answers was successfully saved!")

ebm = ExplainableBoostingClassifier()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

ebm.fit(X_train, y_train)
auc = roc_auc_score(y_val, ebm.predict_proba(X_val)[:, 1])
print(f"AUC: {auc:.2f}")

show(ebm.explain_global())

