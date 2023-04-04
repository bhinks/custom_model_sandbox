import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/10kDiabetes.csv")

df = df.rename(columns={'glyburide.metformin': 'glyburide_metformin', 'glipizide.metformin': 'glipizide_metformin'})

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

features = [
    "race_cat", "gender_cat", "age_cat", "weight_cat", "admission_type_id_cat", "discharge_disposition_id_cat", "admission_source_id_cat", 
    "time_in_hospital", "payer_code_cat", "medical_specialty_cat", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "number_diagnoses", "max_glu_serum_cat", "A1Cresult_cat",
    "metformin_cat", "repaglinide_cat", "nateglinide_cat", "chlorpropamide_cat", "glimepiride_cat", "acetohexamide", "glipizide_cat", "glyburide_cat",
    "tolbutamide_cat", "pioglitazone_cat", "rosiglitazone_cat", "acarbose_cat", "miglitol_cat", "troglitazone", "tolazamide_cat", "examide", "citoglipton",
    "insulin_cat", "glyburide_metformin_cat", "glipizide_metformin_cat", "glimepiride.pioglitazone", "metformin.rosiglitazone", "metformin.pioglitazone",
    "change_cat", "diabetesMed"
]

x = np.array(df[features])
y = np.array(df["readmitted"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1337)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

pickle.dump(model, open("deploy/model.pkl", "wb"))