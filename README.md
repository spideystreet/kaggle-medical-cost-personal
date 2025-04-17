# Medical Insurance Cost Prediction

![Insurance](https://img.shields.io/badge/Insurance-Healthcare-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0.2-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📊 Project Overview

This project analyzes and predicts individual medical insurance costs in the US based on personal characteristics. By identifying key factors influencing healthcare costs, I built a machine learning model that accurately predicts insurance premiums, achieving an R² score of 0.86.

## 🔍 Dataset

Source: [Kaggle - Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance/data)

The dataset contains information on 1,338 insurance beneficiaries with the following features:
- Age
- Sex (male/female)
- BMI (Body Mass Index)
- Number of children/dependents
- Smoking status (yes/no)
- Residential region (northeast, northwest, southeast, southwest)
- Medical charges (target variable)

## 📝 Key Findings

1. **Smoking status** has the most significant impact on insurance costs, especially when combined with BMI
2. **BMI** contributes significantly to cost increases, both directly and in combination with other factors
3. **Age** shows a non-linear relationship with costs, with prices accelerating for older individuals
4. **Geographic region** and **gender** have no impact on insurance pricing
5. The combination of smoking and higher BMI leads to dramatically higher insurance costs

## 🔧 Methodology

### 1. Exploratory Data Analysis
- Analyzed distributions of features and target variable
- Identified key correlations between features and insurance charges
- Visualized the impact of categorical variables on costs

### 2. Data Preprocessing
- Encoded categorical variables (sex, smoker, region)
- Created interaction features (smoker_bmi, age_bmi, smoker_age)
- Added polynomial features to capture non-linear relationships
- Applied log transformation to handle skewed distribution of charges
- Standardized features for model stability

### 3. Model Development
- Implemented and compared multiple regression models
- Linear Regression achieved the best performance with R² of 0.86
- Polynomial Regression did not improve performance (R² of 0.83)
- Performed feature importance analysis to understand key predictors

### 4. Evaluation
- Visualized actual vs. predicted values to assess performance
- Analyzed prediction errors across different demographic groups
- Confirmed model's ability to generalize to unseen data

## 📈 Results

My Linear Regression model successfully captures the key determinants of insurance costs:

| Feature     | Impact on Charges |
|-------------|------------------|
| smoker_bmi  | Very high positive |
| smoker      | High negative (offset by smoker_bmi) |
| age_squared | Moderate positive |
| bmi         | Moderate positive |
| children    | Low positive |
| region      | Minimal |
| sex         | Minimal |

The model performs particularly well for individuals with lower to moderate insurance costs, with some reduced accuracy for very high-cost individuals.

## 💻 Usage

### Prediction Function
```python
def predict_insurance_charge(model, age, sex, bmi, children, smoker, region, scaler=None):
    """
    Predict insurance charges based on personal characteristics
    
    Parameters:
    -----------
    model : trained model object
        The trained regression model
    age : int - Age of individual (18-100)
    sex : str - 'male' or 'female'
    bmi : float - Body Mass Index value
    children : int - Number of dependents
    smoker : str - 'yes' or 'no'
    region : str - 'northeast', 'northwest', 'southeast', or 'southwest'
    scaler : object, optional
        Fitted scaler object if scaling was applied during training
        
    Returns:
    --------
    float - Predicted annual insurance cost in USD
    """
    # Implementation details in src/predict.py
```

### Example
```python
# Load the trained model
import pickle
with open('models/insurance_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']

# Predict insurance cost for a 30-year-old male
cost = predict_insurance_charge(
    model=model,
    age=30, 
    sex='male', 
    bmi=25.0, 
    children=0, 
    smoker='no', 
    region='southeast'
)
print(f"Estimated annual insurance cost: ${cost:.2f}")
```

## 🚀 Future Improvements

- Experiment with more advanced models (Random Forest, Gradient Boosting)
- Develop an interactive web application for insurance cost prediction