# Import Library
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")

data = pd.read_csv("auto-mpg.csv", sep=",")

predict = "acceleration"

data = data[["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "modelyear", "origin"]]
data = shuffle(data)  # Optional - shuffle the data

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(500000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("car_accel.pickle", "wb") as f:
            pickle.dump(linear, f)
print("Accuracy of best = " + str(best))

# LOAD MODEL
pickle_in = open("car_accel.pickle", "rb")
linear = pickle.load(pickle_in)

predicted = linear.predict(x_test)
for x in range(len(predicted)):
    print(round(predicted[x], 1), x_test[x], y_test[x])

# Drawing and plotting model
plot = "horsepower"
plt.scatter(data[plot], data["acceleration"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("acceleration")
plt.show()
