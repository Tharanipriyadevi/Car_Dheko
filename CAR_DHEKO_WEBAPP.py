import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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
    set_background_image("D://Project/Car_Dheko/op/car_photo.jpg")  # Replace with your image file path

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

    # Load the model
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return

    # Load the dataset
    data = pd.read_csv(csv_path)

    # Initialize LabelEncoder and MinMaxScaler
    label_encoders = {}
    scalers = {}
    
    # Create LabelEncoders for categorical features
    categorical_features = ['ft', 'bt', 'transmission', 'model', 'Insurance Validity', 'Seats', 
                            'Engine Displacement', 'Drive Type', 'City']
    for feature in categorical_features:
        if feature in data.columns:
            le = LabelEncoder()
            le.fit(data[feature].astype(str))
            label_encoders[feature] = le
        else:
            st.warning(f"Categorical feature '{feature}' not found in the dataset.")

    # Create MinMaxScaler for numerical features
    numerical_features = ['Mileage', 'Max Power', 'Torque', 'No of Cylinder', 'Length', 
                         'Width', 'Height', 'Kerb Weight', 'Cargo Volume', 'Car_age']

    # Feature Engineering: Calculate km_per_year and drop 'km'
    data['km_per_year'] = data['km'] / data['Car_age'].replace(0, np.nan)
    data = data.drop(columns=['km'])

    # Additional Feature Engineering
    data['power_to_weight'] = data['Max Power'] / data['Kerb Weight']  # Power-to-Weight Ratio
    data['engine_size_per_cylinder'] = data['Engine Displacement'] / data['No of Cylinder'].replace(0, np.nan)  # Size per Cylinder
    data['torque_to_power'] = data['Torque'] / data['Max Power'].replace(0, np.nan)  # Torque-to-Power Ratio
    data['volume'] = data['Length'] * data['Width'] * data['Height']  # Volume of the Car

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
        ft = st.selectbox("Fuel Type", options=sorted(data['ft'].dropna().unique()))      
        bt = st.selectbox("Body Type", options=sorted(data['bt'].dropna().unique()))
        transmission = st.selectbox("Transmission", options=sorted(data['transmission'].dropna().unique()))
        model_input = st.selectbox("Model", options=sorted(data['model'].dropna().unique()))
        Insurance_Validity = st.selectbox("Insurance Validity", options=sorted(data['Insurance Validity'].dropna().unique()))
        Seats = st.selectbox("Seats", options=sorted(data['Seats'].dropna().unique()))
        Engine_Displacement = st.slider("Engine Displacement", min_value=int(data['Engine Displacement'].min()), max_value=int(data['Engine Displacement'].max()), value=int(data['Engine Displacement'].min()))
        Mileage = st.slider("Mileage", min_value=int(data['Mileage'].min()), max_value=int(data['Mileage'].max()), value=int(data['Mileage'].min()))
        Max_Power = st.selectbox("Max Power", options=sorted(data['Max Power'].dropna().unique()))
        Torque = st.selectbox("Torque", options=sorted(data['Torque'].dropna().unique()))
        No_of_Cylinder = st.slider("No of Cylinder", min_value=int(data['No of Cylinder'].min()), max_value=int(data['No of Cylinder'].max()), value=int(data['No of Cylinder'].min()))
        Length = st.slider("Length", min_value=int(data['Length'].min()), max_value=int(data['Length'].max()), value=int(data['Length'].min()))
        Width = st.slider("Width", min_value=int(data['Width'].min()), max_value=int(data['Width'].max()), value=int(data['Width'].min()))
        Height = st.slider("Height", min_value=int(data['Height'].min()), max_value=int(data['Height'].max()), value=int(data['Height'].min()))
        Kerb_Weight = st.selectbox("Kerb Weight", options=sorted(data['Kerb Weight'].dropna().unique()))
        Drive_Type = st.selectbox("Drive Type", options=sorted(data['Drive Type'].dropna().unique()))
        Cargo_Volume = st.selectbox("Cargo Volume", options=sorted(data['Cargo Volume'].dropna().unique()))
        City = st.selectbox("City", options=sorted(data['City'].dropna().unique()))
        Car_age = st.slider("Car Age", min_value=int(data['Car_age'].min()), max_value=int(data['Car_age'].max()), value=int(data['Car_age'].min()))

        # Input for kilometers driven per year
        min_km_per_year = int(data['km_per_year'].min())
        max_km_per_year = int(data['km_per_year'].max())
        km_per_year = st.slider("Kilometers Driven per Year", min_value=min_km_per_year, max_value=max_km_per_year, value=min_km_per_year)

    # Calculate additional features
    power_to_weight = Max_Power / Kerb_Weight
    engine_size_per_cylinder = Engine_Displacement / No_of_Cylinder
    torque_to_power = Torque / Max_Power
    volume = Length * Width * Height

    # Prepare input DataFrame with the transformed features
    input_data = {
        'ft': [ft],
        'bt': [bt],
        'transmission': [transmission],
        'model': [model_input],  # Change this to avoid overwriting the model variable
        'Insurance Validity': [Insurance_Validity],
        'Seats': [Seats],
        'Engine Displacement': [Engine_Displacement],
        'Mileage': [Mileage],
        'Max Power': [Max_Power],
        'Torque': [Torque],
        'No of Cylinder': [No_of_Cylinder],
        'Length': [Length],
        'Width': [Width],
        'Height': [Height],
        'Kerb Weight': [Kerb_Weight],
        'Drive Type': [Drive_Type],
        'Cargo Volume': [Cargo_Volume],
        'City': [City],
        'Car_age': [Car_age],
        'km_per_year': [km_per_year],
        'power_to_weight': [power_to_weight],
        'engine_size_per_cylinder': [engine_size_per_cylinder],
        'torque_to_power': [torque_to_power],
        'volume': [volume]
    }
    input_df = pd.DataFrame(input_data)

    # Perform label encoding for categorical features
    for feature in categorical_features:
        if feature in label_encoders:
            try:
                input_df[feature] = label_encoders[feature].transform(input_df[feature].astype(str))
            except ValueError:
                unseen_labels = set(input_df[feature]) - set(label_encoders[feature].classes_)
                if unseen_labels:
                    # Handle unseen labels by mapping them to the first class or a default value
                    st.warning(f"Unseen labels for {feature}: {unseen_labels}. Mapping to first class.")
                    input_df[feature] = input_df[feature].apply(lambda x: label_encoders[feature].classes_[0] if x not in label_encoders[feature].classes_ else x)
                input_df[feature] = label_encoders[feature].transform(input_df[feature].astype(str))

    # Scale numerical features
    input_df[numerical_features] = scalers['numerical'].transform(input_df[numerical_features])

    # Make predictions
    if st.button('Predict Price'):
        try:
            prediction = model.predict(input_df)
            st.markdown(
                f"<h2 style='color:#f7e5ac;'>Estimated Price: ₹{prediction[0]:,.2f}</h2>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error making prediction: {e}")
   

if __name__ == "__main__":
    main()
