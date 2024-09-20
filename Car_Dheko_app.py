import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import base64

# Function to set a background image
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            color: red;
        }}
        .stSidebar {{
            background-color: black;
            color: red;
        }}
        .stButton button {{
            background-color: red;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Define the main function
def main():
    # Set the background image (provide the path to your image)
    set_background_image("D://Project/Car_Dheko/op/car_photo.jpg")  # Replace 'background.jpg' with your image file name

    # Check if the CSV file exists
    csv_path = "D:/Project/Car_Dheko/op/dropped_car_data_set.csv"
    if not os.path.exists(csv_path):
        st.error(f"File not found: {csv_path}")
        return

    # Load the pre-trained model using pickle
    model_path = "D:/Project/Car_Dheko/op/gradient_boosting_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Load the dataset
    data = pd.read_csv(csv_path)

    # Initialize LabelEncoder and MinMaxScaler
    label_encoders = {}
    scalers = {}

    # Create LabelEncoders for categorical features
    categorical_features = ['bt', 'transmission', 'oem', 'variantName', 'Insurance Validity', 'Engine Displacement', 'Drive Type']
    for feature in categorical_features:
        if feature in data.columns:
            le = LabelEncoder()
            le.fit(data[feature].astype(str))
            label_encoders[feature] = le
        else:
            st.warning(f"Categorical feature '{feature}' not found in the dataset.")

    # Create MinMaxScaler for numerical features
    numerical_features = ['Mileage', 'Car_age', 'km_per_year', 'Engine_displacement_mileage', 'log_km']
    # Compute additional features
    data['km_per_year'] = data['km'] / data['Car_age']
    data['Engine_displacement_mileage'] = data['Engine Displacement'] * data['Mileage']
    data['log_km'] = np.log1p(data['km'])
    data = data.drop(columns=['km'])

    # Ensure all numerical features are in the DataFrame
    missing_features = [feature for feature in numerical_features if feature not in data.columns]
    if missing_features:
        st.error(f"Missing numerical features in the dataset: {', '.join(missing_features)}")
        return
    
    # Initialize MinMaxScaler and fit to the data
    scalers['numerical'] = MinMaxScaler()
    scalers['numerical'].fit(data[numerical_features])

    # Streamlit app interface
    st.title("Car Price Prediction")

    # Sidebar input fields for specific features
    with st.sidebar:
        st.header("Input Features")
        bt = st.selectbox("Body Type", options=sorted(data['bt'].dropna().unique()))
        transmission = st.selectbox("Transmission", options=sorted(data['transmission'].dropna().unique()))
        oem = st.selectbox("OEM", options=sorted(data['oem'].dropna().unique()))
        variantName = st.selectbox("Variant Name", options=sorted(data['variantName'].dropna().unique()))
        Insurance_Validity = st.selectbox("Insurance Validity", options=sorted(data['Insurance Validity'].dropna().unique()))
        Engine_Displacement = st.selectbox("Engine Displacement", options=sorted(data['Engine Displacement'].dropna().unique()))
        Mileage = st.selectbox("Mileage", options=sorted(data['Mileage'].dropna().unique()))
        Drive_Type = st.selectbox("Drive Type", options=sorted(data['Drive Type'].dropna().unique()))
        Car_age = st.selectbox("Car Age", options=sorted(data['Car_age'].dropna().unique()))

        # Input for kilometers driven per year
        min_km_per_year = int(data['km_per_year'].min())
        max_km_per_year = int(data['km_per_year'].max())
        km_per_year = st.slider("Kilometers Driven per Year", min_value=min_km_per_year, max_value=max_km_per_year, value=min_km_per_year)

    # Calculate additional features
    Engine_displacement_mileage = Engine_Displacement * Mileage
    log_km = np.log1p(km_per_year)

    # Prepare input DataFrame with the transformed features
    input_data = {
        'bt': [bt],
        'transmission': [transmission],
        'oem': [oem],
        'variantName': [variantName],
        'Insurance Validity': [Insurance_Validity],
        'Engine Displacement': [Engine_Displacement],
        'Mileage': [Mileage],
        'Drive Type': [Drive_Type],
        'Car_age': [Car_age],
        'km_per_year': [km_per_year],
        'Engine_displacement_mileage': [Engine_displacement_mileage],
        'log_km': [log_km]
    }
    input_df = pd.DataFrame(input_data)

    # Perform label encoding for categorical features
    for feature in categorical_features:
        if feature in label_encoders:
            input_df[feature] = label_encoders[feature].transform(input_df[feature].astype(str))
        else:
            st.warning(f"LabelEncoder for '{feature}' not found.")

    # Perform Min-Max scaling for numerical features
    missing_numerical_features = [feature for feature in numerical_features if feature not in input_df.columns]
    if missing_numerical_features:
        st.error(f"Missing numerical features in input data: {', '.join(missing_numerical_features)}")
        return

    try:
        input_df[numerical_features] = scalers['numerical'].transform(input_df[numerical_features])
    except Exception as e:
        st.error(f"Error scaling numerical features: {e}")
        return

    # Main page prediction button and result display
    if st.button('Predict Price'):
        try:
            prediction = model.predict(input_df)
            st.markdown(
                f"<h2 style='color:#f7e5ac;'>Estimated Price: {prediction[0]:,.2f}</h2>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
