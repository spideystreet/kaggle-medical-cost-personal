import pandas as pd

def predict_insurance_charge(model, age, sex, bmi, children, smoker, region, scaler=None):
    """Function to predict insurance charges for new individuals
    
    Parameters:
    -----------
    model : trained model object
        The trained regression model
    age : int
        Age of the individual
    sex : str
        'male' or 'female'
    bmi : float
        Body Mass Index
    children : int
        Number of children/dependents
    smoker : str
        'yes' or 'no'
    region : str
        'northeast', 'northwest', 'southeast', or 'southwest'
    scaler : object, optional
        Fitted scaler object if scaling was applied during training
        
    Returns:
    --------
    float
        Predicted insurance charge in USD
    """
    # Create a DataFrame with input data
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [1 if sex == 'male' else 0],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == 'yes' else 0],
        'region_northwest': [1 if region == 'northwest' else 0],
        'region_southeast': [1 if region == 'southeast' else 0],
        'region_southwest': [1 if region == 'southwest' else 0]
    })
    
    # Create derived features
    input_data['age_bmi'] = input_data['age'] * input_data['bmi']
    input_data['smoker_age'] = input_data['smoker'] * input_data['age']
    input_data['smoker_bmi'] = input_data['smoker'] * input_data['bmi']
    input_data['bmi_squared'] = input_data['bmi'] ** 2
    input_data['age_squared'] = input_data['age'] ** 2
    
    # Apply scaling if available
    if scaler is not None:
        try:
            input_data_scaled = scaler.transform(input_data)
            input_data = pd.DataFrame(input_data_scaled, columns=input_data.columns)
        except Exception as e:
            print(f"Warning: Could not apply scaling - {e}")
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    return prediction

# Usage example (not executed when imported)
if __name__ == "__main__":
    # This code only runs when the file is executed directly, not when imported
    import pickle
    
    # Load model 
    with open('../models/insurance_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    
    # Try a prediction
    cost = predict_insurance_charge(
        model=model,
        age=30,
        sex='male',
        bmi=25,
        children=0,
        smoker='no',
        region='southeast'
    )
    
    print(f"Predicted cost: ${cost:.2f}")