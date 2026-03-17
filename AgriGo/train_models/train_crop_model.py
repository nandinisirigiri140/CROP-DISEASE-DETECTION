import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# load dataset
df = pd.read_csv("dataset/Crop_recommendation.csv")

# encoders
le_season = LabelEncoder()
le_soil = LabelEncoder()
le_crop = LabelEncoder()

df["Season"] = le_season.fit_transform(df["Season"])
df["Soil"] = le_soil.fit_transform(df["Soil"])
df["Crop"] = le_crop.fit_transform(df["Crop"])

X = df[["Season","Soil"]]
y = df["Crop"]

model = DecisionTreeClassifier()
model.fit(X,y)

# save model
pickle.dump(model,open("crop_model.pkl","wb"))

# save encoders
pickle.dump(le_season,open("season_encoder.pkl","wb"))
pickle.dump(le_soil,open("soil_encoder.pkl","wb"))
pickle.dump(le_crop,open("crop_encoder.pkl","wb"))

print("Crop model created")