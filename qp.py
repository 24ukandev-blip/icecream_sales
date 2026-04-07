"""
Ice Cream Sales Prediction - Optimized for Fast Loading
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Lightweight imports only
st.set_page_config(page_title="🍦 Ice Cream Sales Predictor", layout="wide")
st.title("🍦 Ice Cream Sales Prediction System")
st.markdown("---")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'is_trained' not in st.session_state:
    st.session_state.is_trained = False

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose an option",
    ["📊 1. Train Model", "🎯 2. Single Prediction", "📂 3. Batch Prediction"]
)

# -------------------- HELPER FUNCTIONS (imports inside) --------------------
def create_engineered_features(df):
    """Add engineered features (no heavy imports)"""
    df = df.copy()
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Day_of_Month'] = df['Date'].dt.day
            df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
            def get_season(m):
                if m in [12,1,2]: return 0
                elif m in [3,4,5]: return 1
                elif m in [6,7,8]: return 2
                else: return 3
            df['Season'] = df['Month'].apply(get_season)
        except:
            df['Day_of_Week'] = 0; df['Month'] = 6; df['Day_of_Month'] = 15; df['Is_Weekend'] = 0; df['Season'] = 2
    else:
        df['Day_of_Week'] = 0; df['Month'] = 6; df['Day_of_Month'] = 15; df['Is_Weekend'] = 0; df['Season'] = 2
    df['Temp_Humidity'] = df['Temperature_C'] * df['Humidity_%'] / 100
    df['Temp_Promo'] = df['Temperature_C'] * df['Promotion']
    df['Humidity_Weekend'] = df['Humidity_%'] * df['Weekend']
    df['Price_Promo'] = df['Price_per_Unit'] * df['Promotion']
    median_units = df['Units_Sold'].median() if 'Units_Sold' in df.columns else 50
    df['Units_Sold_Lag1'] = median_units
    df['Units_Sold_MA7'] = median_units
    return df

# -------------------- TRAIN MODEL SECTION --------------------
if option == "📊 1. Train Model":
    st.header("📊 Train Your Ice Cream Sales Model")
    
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded! Shape: {df.shape}")
            with st.expander("Preview Dataset"):
                st.dataframe(df.head())
            
            # Feature engineering
            with st.spinner("Creating features..."):
                df_engineered = create_engineered_features(df)
                q33 = df_engineered['Units_Sold'].quantile(0.33)
                q66 = df_engineered['Units_Sold'].quantile(0.66)
                def categorize(units):
                    if units <= q33: return 'Low'
                    elif units <= q66: return 'Medium'
                    else: return 'High'
                df_engineered['Sales_Category'] = df_engineered['Units_Sold'].apply(categorize)
                st.write("Target Distribution:", df_engineered['Sales_Category'].value_counts().to_dict())
            
            # Prepare features
            drop_cols = ['Date', 'Units_Sold', 'Per_Day_Revenue', 'Monthly_Revenue', 'Yearly_Revenue', 'Sales_Category']
            drop_cols = [c for c in drop_cols if c in df_engineered.columns]
            X = df_engineered.drop(columns=drop_cols)
            y = df_engineered['Sales_Category']
            
            # Import ML libraries only now (heavy imports deferred)
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
            import joblib
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Define column types
            numeric_features = ['Temperature_C', 'Humidity_%', 'Wind_Speed_kmph', 'Price_per_Unit', 
                                'Promotion', 'Day_of_Week', 'Month', 'Day_of_Month', 'Is_Weekend', 'Season',
                                'Temp_Humidity', 'Temp_Promo', 'Humidity_Weekend', 'Price_Promo',
                                'Units_Sold_Lag1', 'Units_Sold_MA7']
            numeric_features = [c for c in numeric_features if c in X.columns]
            categorical_features = ['Weather', 'IceCream_Flavour', 'Variety', 'Store_Location']
            categorical_features = [c for c in categorical_features if c in X.columns]
            binary_features = ['Holiday', 'Weekend']
            binary_features = [c for c in binary_features if c in X.columns]
            
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
                ('bin', 'passthrough', binary_features)
            ])
            
            X_processed = preprocessor.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
            
            # Train models
            st.subheader("🤖 Model Training & Comparison")
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'SVM': SVC(kernel='rbf', random_state=42)
            }
            
            results = []
            best_model = None
            best_acc = 0
            progress_bar = st.progress(0)
            for i, (name, model) in enumerate(models.items()):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results.append({'Model': name, 'Accuracy': f"{acc:.2%}"})
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                progress_bar.progress((i+1)/len(models))
            progress_bar.empty()
            
            results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
            st.table(results_df)
            st.success(f"🏆 Best Model: {results_df.iloc[0]['Model']} with {results_df.iloc[0]['Accuracy']}")
            
            # Confusion Matrix
            y_pred_best = best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_best, labels=['Low','Medium','High'])
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                        xticklabels=['Low','Medium','High'],
                        yticklabels=['Low','Medium','High'], ax=ax)
            ax.set_title(f'Confusion Matrix - {results_df.iloc[0]["Model"]}')
            st.pyplot(fig)
            
            # Save model
            st.session_state.model = best_model
            st.session_state.preprocessor = preprocessor
            st.session_state.feature_cols = list(X.columns)
            st.session_state.is_trained = True
            
            joblib.dump(best_model, 'best_model.pkl')
            joblib.dump(preprocessor, 'preprocessor.pkl')
            st.success("✅ Model saved successfully!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("📂 Please upload your CSV file to start training.")

# -------------------- SINGLE PREDICTION SECTION --------------------
elif option == "🎯 2. Single Prediction":
    st.header("🎯 Predict Sales Category for a Single Day")
    
    if not st.session_state.is_trained:
        st.warning("⚠️ No trained model found. Please go to 'Train Model' first.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input("📅 Date", datetime.now())
                temp = st.slider("🌡️ Temperature (°C)", -5.0, 45.0, 25.0, 0.5)
                humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, 50.0, 1.0)
                wind = st.slider("💨 Wind Speed (km/h)", 0.0, 50.0, 10.0, 0.5)
                weather = st.selectbox("☁️ Weather", ['Sunny','Cloudy','Rainy','Windy'])
                holiday = st.checkbox("🏖️ Holiday")
                weekend = st.checkbox("📆 Weekend")
            with col2:
                flavour = st.selectbox("🍦 Flavour", ['Vanilla','Chocolate','Strawberry','Mango','Butterscotch','Pista'])
                variety = st.selectbox("🥄 Variety", ['Cone','Cup','Stick','Family Pack'])
                location = st.selectbox("📍 Location", ['Urban','Semi-Urban','Rural'])
                price = st.slider("💰 Price per Unit (₹)", 10.0, 150.0, 50.0, 0.5)
                promo = st.selectbox("🎁 Promotion", [0,1], format_func=lambda x: "Yes" if x else "No")
            submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
            
            if submitted:
                input_data = pd.DataFrame([{
                    'Temperature_C': temp, 'Humidity_%': humidity, 'Wind_Speed_kmph': wind,
                    'Weather': weather, 'Holiday': int(holiday), 'Weekend': int(weekend),
                    'IceCream_Flavour': flavour, 'Variety': variety, 'Store_Location': location,
                    'Price_per_Unit': price, 'Promotion': promo, 'Date': date
                }])
                input_data = create_engineered_features(input_data)
                X_input = input_data[st.session_state.feature_cols]
                X_processed = st.session_state.preprocessor.transform(X_input)
                pred = st.session_state.model.predict(X_processed)[0]
                if pred == "High":
                    st.balloons()
                    st.success(f"## 🍦 Predicted: **{pred}** 📈")
                elif pred == "Medium":
                    st.info(f"## 🍦 Predicted: **{pred}** 📊")
                else:
                    st.warning(f"## 🍦 Predicted: **{pred}** 📉")

# -------------------- BATCH PREDICTION SECTION --------------------
elif option == "📂 3. Batch Prediction":
    st.header("📂 Batch Prediction")
    if not st.session_state.is_trained:
        st.warning("⚠️ No trained model found. Please go to 'Train Model' first.")
    else:
        batch_file = st.file_uploader("Upload CSV for prediction", type=['csv'])
        if batch_file is not None:
            try:
                df_batch = pd.read_csv(batch_file)
                st.success(f"✅ Loaded {df_batch.shape[0]} rows")
                df_batch = create_engineered_features(df_batch)
                for col in st.session_state.feature_cols:
                    if col not in df_batch.columns:
                        df_batch[col] = 0
                X_batch = df_batch[st.session_state.feature_cols]
                X_processed = st.session_state.preprocessor.transform(X_batch)
                predictions = st.session_state.model.predict(X_processed)
                df_batch['Predicted_Sales_Category'] = predictions
                st.dataframe(df_batch[['Date','Temperature_C','Predicted_Sales_Category']].head(20))
                csv_output = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv_output, "predictions.csv")
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("🍦 **Ice Cream Sales Prediction System** | Fast Loading Version")