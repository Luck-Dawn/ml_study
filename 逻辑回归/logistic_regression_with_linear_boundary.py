import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression
data = pd.read_csv('data/iris.csv')
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

x_axis = 'petal_length'
y_axis = 'petal_width'