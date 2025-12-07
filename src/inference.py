import pickle
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path='../models/restaurant_model.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return model


def load_feature_names(path='../data/preprocessed/feature_names.pkl'):
    with open(path, 'rb') as f:
        feature_names = pickle.load(f)
    return feature_names


def preprocess_input(input_data, feature_names):
    df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    for key, value in input_data.items():
        if key in df.columns:
            df[key] = value
    
    return df


def predict(model, input_data, feature_names):
    X = preprocess_input(input_data, feature_names)
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0] 
    
    result = {
        'prediction': 'Successful' if prediction == 1 else 'Unsuccessful',
        'success_probability': float(probability),
        'confidence': float(probability)
    }
    
    return result


if __name__ == "__main__":
    # Load model and features
    model = load_model()
    feature_names = load_feature_names()
    
    # Example restaurant input
    sample_restaurant = {
        'online_order': 1,
        'book_table': 1,
        'votes': 500,
        'cost_for_two': 800,
        'location_freq': 1500,
        'city_freq': 3000,
        'cuisine_count': 3,
        'cuisine_north_indian': 1,
        'cuisine_chinese': 1
        # other features default to 0
    }
    
    result = predict(model, sample_restaurant, feature_names)
    
    # Print result
    print("PREDICTION RESULT")
    print("="*60)
    print(f"Prediction: {result['prediction']}")
    print(f"Success Probability: {result['success_probability']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")