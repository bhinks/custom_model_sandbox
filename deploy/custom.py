"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd

def transform(data, model):
    data = data.drop(['readmitted'], axis='columns',errors='ignore')
    data = data.rename(columns={
        "glyburide.metformin": "glyburide_metformin", 
        "glipizide.metformin": "glipizide_metformin",
        "glimepiride.pioglitazone": "glimepiride_pioglitazone",
        "metformin.rosiglitazone": "metformin_rosiglitazone",
        "metformin.pioglitazone": "metformin_pioglitazone"
    })
    data = data.drop(["diag_1_desc", "diag_2_desc", "diag_3_desc"], axis=1)

    return data

def score(data, model, **kwargs):
    output = model.predict_proba(data)
    predictions = pd.DataFrame(output, columns=['False','True'])

    return predictions
