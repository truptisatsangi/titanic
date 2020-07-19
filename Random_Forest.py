import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
train_data['label'] = 0

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data['label'] = 1

concat_df = pd.concat([train_data, test_data])
concat_df = pd.get_dummies(concat_df)
train_data = concat_df[concat_df['label']==0]
test_data = concat_df[concat_df['label']==1]
passengerId = test_data['PassengerId']

train_data = train_data.drop('label', axis =1 )
test_data = test_data.drop('label', axis =1)

y = train_data['Survived']
X = train_data.drop('Survived',axis =1)
test_x= test_data.drop('Survived',axis =1)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
X = pd.DataFrame(imputer.fit_transform(X))
test_x = pd.DataFrame(imputer.transform(test_x))
passengerId = test_data['PassengerId']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train_x, val_x, train_y, val_y = train_test_split(X,y,random_state =1)
rf_model = RandomForestClassifier(n_estimators = 150, random_state = 1)
rf_model.fit(train_x, train_y)
pred_val = rf_model.predict(val_x)

mae = mean_absolute_error(val_y, pred_val)
# mae
rf_full_model = RandomForestClassifier(n_estimators = 80, random_state = 1)
rf_full_model.fit(X,y)
Survived = rf_full_model.predict(test_x)

df  = pd.DataFrame()
df['PassengerId'] =passengerId
df['Survived'] = Survived

df = df.astype(int)

df.to_csv('submission.csv',index = False)
df
