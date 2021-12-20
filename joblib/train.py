import pandas as pd
import joblib
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names = ['preg','plas','pres','skin', 'test','mass', 'pedi','age','class']
df = pd.read_csv(url, names= names)
print(df)
print(df.info())
X = df.iloc[:, 0:8]
y = df.iloc[: , 8]
print(X)
print(y)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.20, random_state = 101)
model = LogisticRegression()
model.fit(X_train, y_train)
print('[info] model is trained')
result = model.score(X_test, y_test)
print(f'accuracy of the model is {result}')
joblib.dump(model, 'dib_79.pkl')