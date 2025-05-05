#  Stroke Prediction Project

##  Project Overview
This project uses machine learning to predict the likelihood of a stroke based on patient health attributes. The model was developed using a dataset containing demographic and health information and aims to assist healthcare professionals in identifying high-risk patients for early intervention.

##  Objectives
- Explore and visualize the dataset (EDA)
- Perform statistical inference to validate patterns and relationships
- Train and evaluate multiple machine learning models
- Use hyperparameter tuning (GridSearchCV & BayesSearchCV)
- Build a final voting ensemble model
- Apply threshold tuning to optimize recall and F1-score
- Deploy the model using a Streamlit web application

## Dataset
The dataset includes features such as:
- Gender, Age
- Hypertension, Heart Disease
- Marital Status, Residence Type
- Work Type, Smoking Status
- Average Glucose Level, BMI
- Stroke outcome (target variable)

##  Models Used
- Logistic Regression (tuned)
- Random Forest (Bayesian tuned)
- XGBoost (Bayesian tuned)

These were combined into a soft voting ensemble, which was then optimized using threshold tuning.

##  Evaluation Metrics
- Weighted Accuracy, Precision, Recall, F1-Score
- AUC (Area Under the ROC Curve)
- Confusion Matrix (analyzed before and after threshold tuning)

Final model achieved:
- **Recall**: ~83% at tuned threshold (0.21)
- **AUC**: 0.8460

##  Deployment
The model was deployed using **Streamlit**, a browser-based interactive web app. The app allows users to input patient data and returns:
- A stroke risk prediction (High/Low)
- Probability of stroke

###  Run the App Locally:
```bash
streamlit run app.py
```

Ensure that `stroke_prediction_pipeline.pkl` and `app.py` are in the same folder.

##  Environment Setup
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

##  Files Included
- `stroke_prediction.ipynb`: Full exploratory analysis, modeling, and explanations
- `stroke_prediction_pipeline.pkl`: Serialized model + metadata
- `app.py`: Streamlit web app
- `requirements.txt`: Dependencies for training and deployment

##  Conclusion
This project demonstrates how to go from raw medical data to a fully deployed machine learning model. It balances model performance with clinical interpretability and is deployed using a browser-based web interface.

##  Suggestions for Improvement
- Integrate additional clinical features if available
- Experiment with other ensemble techniques (stacking, boosting)
- Calibrate probabilities for clinical thresholds
- Deploy using a cloud-based solution (e.g., Heroku, AWS Lambda)

## Contributors
- MICHAEL BOND © 2025
  - bondpapi@yahoo.com
  - https://github.com/bondpapi
  - https://www.hackerrank.com/profile/bondpapi
