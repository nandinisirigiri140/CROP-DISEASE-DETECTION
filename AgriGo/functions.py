import os
import pickle
import numpy as np
import tensorflow as tf
import tf_keras as tfk
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# This fix handles the 'int() argument must be... not list' error
class FixedBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        # Convert axis=[3] (old format) to axis=3 (new format)
        if 'axis' in kwargs and isinstance(kwargs['axis'], list):
            kwargs['axis'] = kwargs['axis'][0]
        super().__init__(**kwargs)

# This fix handles the 'groups' parameter error
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

def get_model(path):
    if not os.path.exists(path):
        print(f"CRITICAL ERROR: Model file not found at: {path}")
        return None

    try:
        # We use tfk (tf_keras) because it is the official bridge 
        # for Keras 2 models running on Python 3.12/Keras 3 environments.
        return tfk.models.load_model(path, compile=False)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model at {path}. Error: {e}")
        return None

def img_predict(path, crop):
    # 1. Define model path and load it
    # Note: Using your dictionary key 'patato' if that's what's passed from app.py
    model_path = os.path.join(BASE_DIR, 'models', 'DL_models', f'{crop}_model.keras')
    print(f"Attempting to load: {model_path}") # Debugging line
    
    model = get_model(model_path)

    # 2. Safety check: If model is None, don't try to predict
    if model is None:
        return "Error: Model could not be loaded. Check your terminal for path details."

   # 3. Image Preprocessing
    try:
        data = load_img(path, target_size=(224, 224)) 
        data = np.asarray(data)
        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        data = np.expand_dims(data, axis=0) 
        # Normalize to [0, 1]
        data = data.astype('float32') / 255.0

        # 4. Prediction
        # Use the tf_keras model to predict
        preds = model.predict(data, verbose=0)
        
        # 5. Class Logic
        if len(crop_diseases_classes[crop]) > 2:
            predicted = np.argmax(preds[0])
        else:
            p = preds[0]
            if p.shape[-1] > 1:
                predicted = np.argmax(p)
            else:
                predicted = int(np.round(p)[0])

        return predicted
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error during prediction."

# ================== NEW DISEASE INFO SECTION ==================

disease_info = {
    "Leaf_scorch": {
        "treatment": "Apply copper-based fungicide and remove infected leaves.",
        "precautions": "Avoid overhead irrigation and maintain proper spacing."
    },
    "Early_blight": {
        "treatment": "Spray Mancozeb or Chlorothalonil fungicide every 7 days.",
        "precautions": "Practice crop rotation and remove infected plant debris."
    },
    "Late_blight": {
        "treatment": "Apply Metalaxyl-based fungicide immediately.",
        "precautions": "Avoid water stagnation and use resistant varieties."
    },
    "Cercospora_leaf_spot Gray_leaf_spot": {
        "treatment": "Use recommended fungicide such as Azoxystrobin.",
        "precautions": "Ensure good air circulation and remove infected leaves."
    },
    "Common_rust_": {
        "treatment": "Apply sulfur-based fungicide.",
        "precautions": "Avoid overcrowding and monitor regularly."
    },
    "Northern_Leaf_Blight": {
        "treatment": "Use appropriate fungicide like Propiconazole.",
        "precautions": "Rotate crops and use resistant hybrids."
    },
    "Apple_scab": {
        "treatment": "Spray Captan or Myclobutanil fungicide.",
        "precautions": "Prune trees for air circulation."
    },
    "Black_rot": {
        "treatment": "Apply fungicide and remove infected fruits.",
        "precautions": "Sanitize tools and remove fallen leaves."
    },
    "Cedar_apple_rust": {
        "treatment": "Use protective fungicide sprays.",
        "precautions": "Remove nearby cedar trees if possible."
    },
    "Powdery_mildew": {
        "treatment": "Spray sulfur or potassium bicarbonate solution.",
        "precautions": "Ensure sunlight exposure and reduce humidity."
    },
    "Esca_(Black_Measles)": {
        "treatment": "Remove infected vines.",
        "precautions": "Avoid pruning during wet conditions."
    },
    "Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "treatment": "Apply fungicide such as Mancozeb.",
        "precautions": "Maintain proper drainage."
    },
    "Bacterial_spot": {
        "treatment": "Spray copper-based bactericide.",
        "precautions": "Avoid working in wet fields."
    },
    "Leaf_Mold": {
        "treatment": "Apply fungicide and improve ventilation.",
        "precautions": "Avoid excess humidity."
    },
    "Septoria_leaf_spot": {
        "treatment": "Use Chlorothalonil spray.",
        "precautions": "Remove infected leaves immediately."
    },
    "Spider_mites Two-spotted_spider_mite": {
        "treatment": "Use insecticidal soap or neem oil.",
        "precautions": "Maintain proper humidity."
    },
    "Target_Spot": {
        "treatment": "Apply recommended fungicide.",
        "precautions": "Avoid water splash on leaves."
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "treatment": "Remove infected plants immediately.",
        "precautions": "Control whiteflies using insecticides."
    },
    "Tomato_mosaic_virus": {
        "treatment": "Remove infected plants.",
        "precautions": "Disinfect tools and avoid tobacco contact."
    },
    "healthy": {
        "treatment": "No treatment required. Crop is healthy.",
        "precautions": "Maintain proper irrigation and balanced fertilization."
    }
}


def get_diseases_classes(crop, prediction):
    # Check if prediction is an error message (string) instead of a number
    if isinstance(prediction, str):
        return {
            "disease": "Error",
            "treatment": prediction, # Show the error message (e.g., "Model not found")
            "precautions": "Please check your model folder structure."
        }

    try:
        crop_classes = crop_diseases_classes[crop]
        # Ensure prediction is an integer index
        disease_name = crop_classes[int(prediction)][1]

        info = disease_info.get(disease_name, {
            "treatment": "No treatment information available.",
            "precautions": "No precaution information available."
        })

        return {
            "disease": disease_name.replace("_", " "),
            "treatment": info["treatment"],
            "precautions": info["precautions"]
        }
    except Exception as e:
        return {
            "disease": "Unknown",
            "treatment": f"Error retrieving info: {e}",
            "precautions": "Check dictionary mapping."
        }


# ================== UPDATED CROP RECOMMENDATION ==================

def get_crop_recommendation(season, soil_type):

    model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_model.pkl')
    season_encoder_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'season_encoder.pkl')
    soil_encoder_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'soil_encoder.pkl')
    crop_encoder_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_encoder.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(season_encoder_path, 'rb') as f:
        le_season = pickle.load(f)

    with open(soil_encoder_path, 'rb') as f:
        le_soil = pickle.load(f)

    with open(crop_encoder_path, 'rb') as f:
        le_crop = pickle.load(f)

    season_encoded = le_season.transform([season])[0]
    soil_encoded = le_soil.transform([soil_type])[0]

    input_data = np.array([[season_encoded, soil_encoded]])

    prediction = model.predict(input_data)

    crop = le_crop.inverse_transform(prediction)

    return crop[0]


# ================== UPDATED FERTILIZER RECOMMENDATION ==================

def get_fertilizer_recommendation(soil_type, crop_type):
    model_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_model.pkl')
    soil_encoder_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'soil_encoder.pkl')
    crop_encoder_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'crop_encoder.pkl')
    fertilizer_encoder_path = os.path.join(BASE_DIR, 'models', 'ML_models', 'fertilizer_encoder.pkl')

    try:
        with open(model_path, 'rb') as f: model = pickle.load(f)
        with open(soil_encoder_path, 'rb') as f: le_soil = pickle.load(f)
        with open(crop_encoder_path, 'rb') as f: le_crop = pickle.load(f)
        with open(fertilizer_encoder_path, 'rb') as f: le_fertilizer = pickle.load(f)

        # --- THE FIX: SAFETY CHECK ---
        # 1. Clean the input
        crop_type = str(crop_type).strip()
        soil_type = str(soil_type).strip()

        # 2. Check for "Paddy" in the encoder's memory
        if crop_type not in le_crop.classes_:
            return f"Error: '{crop_type}' not in Model. Encoder knows: {list(le_crop.classes_)}"
        
        if soil_type not in le_soil.classes_:
            return f"Error: '{soil_type}' not in Model. Encoder knows: {list(le_soil.classes_)}"

        # 3. Proceed with prediction
        soil_encoded = le_soil.transform([soil_type])[0]
        crop_encoded = le_crop.transform([crop_type])[0]
        
        input_data = np.array([[soil_encoded, crop_encoded]])
        prediction = model.predict(input_data)
        fertilizer = le_fertilizer.inverse_transform(prediction)

        return fertilizer[0]

    except Exception as e:
        return f"System Error: {str(e)}"
# ================== ORIGINAL DATA (UNCHANGED) ==================

crop_diseases_classes = {
    'strawberry': [(0, 'Leaf_scorch'), (1, 'healthy')],
    'patato': [(0, 'Early_blight'), (1, 'Late_blight'), (2, 'healthy')],
    'corn': [(0, 'Cercospora_leaf_spot Gray_leaf_spot'), (1, 'Common_rust_'),
             (2, 'Northern_Leaf_Blight'), (3, 'healthy')],
    'apple': [(0, 'Apple_scab'), (1, 'Black_rot'),
              (2, 'Cedar_apple_rust'), (3, 'healthy')],
    'cherry': [(0, 'Powdery_mildew'), (1, 'healthy')],
    'grape': [(0, 'Black_rot'), (1, 'Esca_(Black_Measles)'),
              (2, 'Leaf_blight_(Isariopsis_Leaf_Spot)'), (3, 'healthy')],
    'peach': [(0, 'Bacterial_spot'), (1, 'healthy')],
    'pepper': [(0, 'Bacterial_spot'), (1, 'healthy')],
    'tomato': [(0, 'Bacterial_spot'), (1, 'Early_blight'),
               (2, 'Late_blight'), (3, 'Leaf_Mold'),
               (4, 'Septoria_leaf_spot'),
               (5, 'Spider_mites Two-spotted_spider_mite'),
               (6, 'Target_Spot'),
               (7, 'Tomato_Yellow_Leaf_Curl_Virus'),
               (8, 'Tomato_mosaic_virus'),
               (9, 'healthy')]
}

crop_list = list(crop_diseases_classes.keys())

soil_types = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
Crop_types = ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets',
              'Oil seeds', 'Paddy', 'Pulses', 'Sugarcane',
              'Tobacco', 'Wheat']

fertilizer_classes = ['10-26-26', '14-35-14', '17-17-17',
                      '20-20', '28-28', 'DAP', 'Urea']