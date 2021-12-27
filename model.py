import pandas as pd
import numpy as np
df = pd.read_csv("Advertising.csv")
X = df.drop('sales',axis=1)
y = df['sales']
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.metrics import mean_squared_error
model = LinearRegression()
model.fit(X_train,y_train)
import pickle
pickle.dump(model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))