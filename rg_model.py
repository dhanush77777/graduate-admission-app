import pandas as pd
import pickle
df=pd.read_csv(r'C:/Users/SAIDHANUSH/Admission_Predict.csv')

x=df.iloc[:,[1,2,3,4,5,6]]
y=df.iloc[:,[8]]


from sklearn.tree import DecisionTreeRegressor
tr=DecisionTreeRegressor()

tr.fit(x,y)

p=tr.predict(x)

pickle.dump(tr,open('rg_model.pkl','wb'))

from sklearn.metrics import r2_score
print(r2_score(y,p))