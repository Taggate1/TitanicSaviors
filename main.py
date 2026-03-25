from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Read in the data given
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data["Survived"]

# Define the features
features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked", "Fare"]
#X = pd.get_dummies(train_data[features])

X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.2, random_state=1)
X1 = pd.get_dummies(X_train[features]) # X1 is train data
X2 = pd.get_dummies(X_val[features]) # X2 is validation
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

# initiate the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1) # amount of decision trees

X_test = pd.get_dummies(test_data[features])
model.fit(X1, y_train)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('final_answers.csv', index=False)
print("Your answers was successfully saved!")

# Global feature importance (built-in)

importances = pd.Series(model.feature_importances_, index=X_test.columns) # sort by importance in model
importances.sort_values().plot(kind='barh')
plt.title("Feature Importances")
plt.xlabel("Mean absolute score (Weighted)")
plt.tight_layout()
plt.show()

A = X_val["Sex"] # gender
Y = y_val # survived or not
D = model.predict(X2) # algorithm prediction on survival

male_mask = (A == "male")
female_mask = (A == "female")

D_a = D[male_mask]
Y_a = Y[male_mask]

TP_a = np.sum((D_a == 1) & (Y_a == 1))
FP_a = np.sum((D_a == 1) & (Y_a == 0))
TN_a = np.sum((D_a == 0) & (Y_a == 0))
FN_a = np.sum((D_a == 0) & (Y_a == 1))

D_b = D[female_mask]
Y_b = Y[female_mask]

TP_b = np.sum((D_b == 1) & (Y_b == 1))
FP_b = np.sum((D_b == 1) & (Y_b == 0))
TN_b = np.sum((D_b == 0) & (Y_b == 0))
FN_b = np.sum((D_b == 0) & (Y_b == 1))

TPR_a = TP_a / (TP_a + FN_a)
TPR_b = TP_b / (TP_b + FN_b)

FPR_a = FP_a / (FP_a + TN_a)
FPR_b = FP_b / (FP_b + TN_b)

FNR_a = FN_a / (FN_a + TP_a)
FNR_b = FN_b / (FN_b + TP_b)

print("Equal opportunity\n",TPR_a)
print(TPR_b, "\n\nEqualized error rate")

print(FPR_a)
print(FPR_b, "\n")

print(FNR_a)
print(FNR_b)