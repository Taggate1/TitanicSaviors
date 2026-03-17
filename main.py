import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Read in the data given
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


y = train_data["Survived"]

# Define the features
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# initiate the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('final_answers.csv', index=False)
print("Your answers was successfully saved!")

