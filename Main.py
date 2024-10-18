import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

house = pd.read_csv("housing.csv")
y = house['price']
x = house.drop(['price', 'prefarea', 'hotwaterheating'], axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
ordinal_encoder = OrdinalEncoder()
Cat_col = [cname for cname in x.columns if x[cname].dtype == 'object']

label_x_train = x_train.copy()
label_x_valid = x_valid.copy()

label_x_train[Cat_col] = ordinal_encoder.fit_transform(x_train[Cat_col])
label_x_valid[Cat_col] = ordinal_encoder.transform(x_valid[Cat_col])

my_model = XGBRegressor(n_estimators=500, early_stopping_rounds=5)
my_model.fit(label_x_train, y_train, eval_set=[(label_x_valid, y_valid)], verbose=False)
pred = my_model.predict(label_x_valid)
print(mean_absolute_error(y_valid, pred))

# 811090.280995935
