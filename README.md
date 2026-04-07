🍦 Ice Cream Sales Prediction
=============================

Predict ice cream sales category (Low/Medium/High) using weather, promotion, and store data.

Quick Start
-----------

1. Install dependencies:
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib

2. Run the app:
   streamlit run app.py

Files
-----
- app.py – Main Streamlit application
- best_model.pkl – Saved best model (created after training)
- preprocessor.pkl – Saved preprocessor

Dataset required columns
------------------------
Date, Temperature_C, Humidity_%, Weather, Wind_Speed_kmph, Holiday, Weekend,
IceCream_Flavour, Variety, Store_Location, Price_per_Unit, Promotion, Units_Sold

Features
--------
- Train 5 ML models & compare accuracy
- Feature engineering (date extraction, interactions)
- Single prediction with sliders & dropdowns
- Batch prediction via CSV upload
- Download trained model

Screenshots for project document
--------------------------------
1. Data preprocessing output
2. Model comparison table
3. Accuracy results
4. Confusion matrix
5. Best model selection
6. GUI interface
7. GUI prediction results

Made with Streamlit 🍦
