import tensorflow as tf
import tf_keras as keras
import os

model_folder = "models/DL_models"

for file in os.listdir(model_folder):
    if file.endswith(".h5"):
        path = os.path.join(model_folder, file)

        print("Converting:", file)

        model = keras.models.load_model(path, compile=False)

        new_name = file.replace(".h5", ".keras")
        new_path = os.path.join(model_folder, new_name)

        model.save(new_path)

print("All models converted!")