"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd

def transform(data, model):
    data = data.drop(['readmitted'], axis='columns',errors='ignore')
    return data

def score(data, model, **kwargs):
    output = model.predict_proba(data)
    predictions = pd.DataFrame(output, columns=['False','True'])

    return predictions
