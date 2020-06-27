import pandas as pd
from sklearn.linear_model import Lasso
import pickle



dataset=pd.read_csv('Jan_2019.csv')
X=dataset.iloc[0:31, 2:6]
y=dataset.iloc[0:31, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_jan.pkl','wb'))


dataset=pd.read_csv('Feb_2019.csv')
X=dataset.iloc[0:28, 2:6]
y=dataset.iloc[0:28, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_feb.pkl','wb'))


dataset=pd.read_csv('Mar_2019.csv')
X=dataset.iloc[0:31, 2:6]
y=dataset.iloc[0:31, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_mar.pkl','wb'))


dataset=pd.read_csv('Apr_2019.csv')
X=dataset.iloc[0:30, 2:6]
y=dataset.iloc[0:30, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_apr.pkl','wb'))


dataset=pd.read_csv('May_2019.csv')
X=dataset.iloc[0:31, 2:6]
y=dataset.iloc[0:31, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_may.pkl','wb'))


dataset=pd.read_csv('Jun_2019.csv')
X=dataset.iloc[0:30, 2:6]
y=dataset.iloc[0:30, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_jun.pkl','wb'))


dataset=pd.read_csv('Jul_2019.csv')
X=dataset.iloc[0:31, 2:6]
y=dataset.iloc[0:31, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_jul.pkl','wb'))


dataset=pd.read_csv('Aug_2019.csv')
X=dataset.iloc[0:31, 2:6]
y=dataset.iloc[0:31, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_aug.pkl','wb'))


dataset=pd.read_csv('Sep_2019.csv')
X=dataset.iloc[0:30, 2:6]
y=dataset.iloc[0:30, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_sep.pkl','wb'))


dataset=pd.read_csv('Oct_2019.csv')
X=dataset.iloc[0:31, 2:6]
y=dataset.iloc[0:31, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_oct.pkl','wb'))


dataset=pd.read_csv('Nov_2019.csv')
X=dataset.iloc[0:30, 2:6]
y=dataset.iloc[0:30, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_nov.pkl','wb'))


dataset=pd.read_csv('Dec_2019.csv')
X=dataset.iloc[0:31, 2:6]
y=dataset.iloc[0:31, 6:10]
lassoreg = Lasso(alpha=0.1)
lassoreg.fit(X,y)
pickle.dump(lassoreg, open('model_dec.pkl','wb'))







