import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

df = pd.read_csv('student_data.csv')
X = df[['math_score', 'reading_score', 'writing_score']]
y_class = (df['math_score'] >= 40).astype(int)
y_reg = df['math_score']

clf = RandomForestClassifier()
clf.fit(X, y_class)
joblib.dump(clf, 'classifier.pkl')

reg = LinearRegression()
reg.fit(X, y_reg)
joblib.dump(reg, 'regressor.pkl')
