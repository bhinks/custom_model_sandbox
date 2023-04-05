"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import json

def transform(data, model):
    data = data.drop(["readmitted"], axis='columns',errors='ignore')
    with open("pandas_schema.json", "r") as f:
        schema = f.readlines()[0]
        pandas_schema = json.loads(schema)
    columns = set(data.columns)
    orig_columns = set(pandas_schema.keys())
    for c in orig_columns.difference(columns):
        del pandas_schema[c]
    data = data.astype(model.pandas_schema)
    return data