import json
import numpy as np
import pandas as pd
import pickle

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/10kDiabetes.csv")

df = df.rename(columns={
    "glyburide.metformin": "glyburide_metformin", 
    "glipizide.metformin": "glipizide_metformin",
    "glimepiride.pioglitazone": "glimepiride_pioglitazone",
    "metformin.rosiglitazone": "metformin_rosiglitazone",
    "metformin.pioglitazone": "metformin_pioglitazone"
})

with open("deploy/pandas_schema.json", "r") as f:
    schema = f.readlines()[0]
    pandas_schema = json.loads(schema)
columns = set(df.columns)
orig_columns = set(pandas_schema.keys())
for c in orig_columns.difference(columns):
    del pandas_schema[c]
df = df.astype(pandas_schema)

x = df.drop(["readmitted"], axis=1)
y = df["readmitted"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.18, random_state=1337)

ohe_features_to_encode = x_train.columns[x_train.dtypes==object].tolist()

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

sim_num_features_to_encode = list(x_train.select_dtypes(include = ['float64', 'int64']).columns)
sim_num = SimpleImputer(missing_values = np.nan, strategy = "mean")

col_trans = make_column_transformer((categorical_transformer, ohe_features_to_encode), (sim_num, sim_num_features_to_encode), remainder = "passthrough")

model = RandomForestClassifier(min_samples_leaf=50, n_estimators=150, bootstrap=True, oob_score=True, n_jobs=-1, random_state=1337, max_features='auto')

pipe = make_pipeline(col_trans, model)
pipe.fit(x_train,y_train)

pickle.dump(pipe, open("deploy/model.pkl", "wb"))
