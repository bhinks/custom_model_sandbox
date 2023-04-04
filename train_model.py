import numpy as np
import pandas as pd
import pickle

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/10kDiabetes.csv")

x = df.drop(["readmitted"], axis=1)
y = df["readmitted"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1337)

ohe_features_to_encode = x_train.columns[x_train.dtypes==object].tolist()
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

sim_num_features_to_encode = list(x_train.select_dtypes(include = ['float64', 'int64']).columns)
sim_num = SimpleImputer(missing_values = np.nan, strategy = 'mean')

col_trans = make_column_transformer((categorical_transformer, ohe_features_to_encode), (sim_num, sim_num_features_to_encode), remainder = "passthrough")

model = LogisticRegression()

pipe = make_pipeline(col_trans, model)
pipe.fit(x_train,y_train)

pipe.fit(x_train, y_train)

pickle.dump(pipe, open("deploy/model.pkl", "wb"))