import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

#implementing training data
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#storing the best fit line using the linear regression model
linear = linear_model.LinearRegression()

#uses the x and y data to train the AI and determine a line of best fit for the data
linear.fit(x_train, y_train)

#stores the test results based of the training data
accuracy = linear.score(x_test, y_test)
print(accuracy)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

#showing input and output data after going through the predictor
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
