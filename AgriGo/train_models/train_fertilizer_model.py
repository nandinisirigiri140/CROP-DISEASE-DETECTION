import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv("dataset/Fertilizer Prediction.csv")

le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df["Soil"] = le_soil.fit_transform(df["Soil"])
df["Crop"] = le_crop.fit_transform(df["Crop"])
df["Fertilizer"] = le_fert.fit_transform(df["Fertilizer"])

X = df[["Soil","Crop"]]
y = df["Fertilizer"]

model = DecisionTreeClassifier()
model.fit(X,y)

pickle.dump(model,open("fertilizer_model.pkl","wb"))

pickle.dump(le_soil,open("soil_encoder_f.pkl","wb"))
pickle.dump(le_crop,open("crop_encoder_f.pkl","wb"))
pickle.dump(le_fert,open("fertilizer_encoder.pkl","wb"))

print("Fertilizer model created")