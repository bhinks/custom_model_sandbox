"""
Copyright 2021 DataRobot, Inc. and its affiliates.
All rights reserved.
This is proprietary source code of DataRobot, Inc. and its affiliates.
Released under the terms of DataRobot Tool and Utility Agreement.
"""
import pandas as pd
import numpy as np


def transform(data, model):
    """
    Note: This hook may not have to be implemented for your model.
    In this case implemented for the model used in the example.

    Modify this method to add data transformation before scoring calls. For example, this can be
    used to implement one-hot encoding for models that don't include it on their own.

    Parameters
    ----------
    data: pd.DataFrame
    model: object, the deserialized model

    Returns
    -------
    pd.DataFrame
    """
    # Execute any steps you need to do before scoring
    # Remove target columns if they're in the dataset
    df = data.rename(columns={
        "glyburide.metformin": "glyburide_metformin", 
        "glipizide.metformin": "glipizide_metformin",
        "glimepiride.pioglitazone": "glimepiride_pioglitazone",
        "metformin.rosiglitazone": "metformin_rosiglitazone",
        "metformin.pioglitazone": "metformin_pioglitazone"
    })
    for target_col in ["readmitted", "diag_1", "diag_2", "diag_3", "diag_1_desc", "diag_2_desc", "diag_3_desc"]:
        if target_col in df:
            df.pop(target_col)
    df = df.fillna(0)

    df.replace("None", np.nan, inplace=True)
    df.replace("No", 0, inplace=True)
    df.replace("Yes", 1, inplace=True)

    df.race = pd.Categorical(df.race)
    df["race_cat"] = df.race.cat.codes

    df.gender = pd.Categorical(df.gender)
    df["gender_cat"] = df.gender.cat.codes

    df.age = pd.Categorical(df.age)
    df["age_cat"] = df.age.cat.codes

    df.weight = pd.Categorical(df.weight)
    df["weight_cat"] = df.weight.cat.codes

    df.admission_type_id = pd.Categorical(df.admission_type_id)
    df["admission_type_id_cat"] = df.admission_type_id.cat.codes

    df.discharge_disposition_id = pd.Categorical(df.discharge_disposition_id)
    df["discharge_disposition_id_cat"] = df.discharge_disposition_id.cat.codes

    df.admission_source_id = pd.Categorical(df.admission_source_id)
    df["admission_source_id_cat"] = df.admission_source_id.cat.codes

    df.payer_code = pd.Categorical(df.payer_code)
    df["payer_code_cat"] = df.payer_code.cat.codes

    df.medical_specialty = pd.Categorical(df.medical_specialty)
    df["medical_specialty_cat"] = df.medical_specialty.cat.codes

    df.A1Cresult = pd.Categorical(df.A1Cresult)
    df["A1Cresult_cat"] = df.A1Cresult.cat.codes

    df.insulin = pd.Categorical(df.insulin)
    df["insulin_cat"] = df.insulin.cat.codes

    df.change = pd.Categorical(df.change)
    df["change_cat"] = df.change.cat.codes

    df.max_glu_serum = pd.Categorical(df.max_glu_serum)
    df["max_glu_serum_cat"] = df.max_glu_serum.cat.codes

    df.metformin = pd.Categorical(df.metformin)
    df["metformin_cat"] = df.metformin.cat.codes

    df.repaglinide = pd.Categorical(df.repaglinide)
    df["repaglinide_cat"] = df.repaglinide.cat.codes

    df.nateglinide = pd.Categorical(df.nateglinide)
    df["nateglinide_cat"] = df.nateglinide.cat.codes

    df.chlorpropamide = pd.Categorical(df.chlorpropamide)
    df["chlorpropamide_cat"] = df.chlorpropamide.cat.codes

    df.glimepiride = pd.Categorical(df.glimepiride)
    df["glimepiride_cat"] = df.glimepiride.cat.codes

    df.glipizide = pd.Categorical(df.glipizide)
    df["glipizide_cat"] = df.glipizide.cat.codes

    df.glyburide = pd.Categorical(df.glyburide)
    df["glyburide_cat"] = df.glyburide.cat.codes

    df.tolbutamide = pd.Categorical(df.tolbutamide)
    df["tolbutamide_cat"] = df.tolbutamide.cat.codes

    df.pioglitazone = pd.Categorical(df.pioglitazone)
    df["pioglitazone_cat"] = df.pioglitazone.cat.codes

    df.rosiglitazone = pd.Categorical(df.rosiglitazone)
    df["rosiglitazone_cat"] = df.rosiglitazone.cat.codes

    df.acarbose = pd.Categorical(df.acarbose)
    df["acarbose_cat"] = df.acarbose.cat.codes

    df.miglitol = pd.Categorical(df.miglitol)
    df["miglitol_cat"] = df.miglitol.cat.codes

    df.tolazamide = pd.Categorical(df.tolazamide)
    df["tolazamide_cat"] = df.tolazamide.cat.codes

    df.glyburide_metformin = pd.Categorical(df.glyburide_metformin)
    df["glyburide_metformin_cat"] = df.glyburide_metformin.cat.codes

    df.glipizide_metformin = pd.Categorical(df.glipizide_metformin)
    df["glipizide_metformin_cat"] = df.glipizide_metformin.cat.codes

    return df
