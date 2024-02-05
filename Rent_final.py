import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# Label encodings
type_encoding = {'RK1': 0, 'BHK1': 1, 'BHK2': 2, 'BHK3': 3, 'BHK4': 4, 'BHK4PLUS': 5}
lease_type_encoding = {'ANYONE': 1, 'FAMILY': 2, 'BACHELOR': 0, 'COMPANY': 3}
furnishing_encoding = {'NOT_FURNISHED': 0, 'SEMI_FURNISHED': 1, 'FULLY_FURNISHED': 2}
parking_encoding = {'TWO_WHEELER': 0, 'NONE': 1, 'FOUR_WHEELER': 2, 'BOTH': 3}
facing_encoding = {'N': 0, 'E': 1, 'W': 2, 'NE': 3, 'NW': 4, 'S': 5, 'SE': 6, 'SW': 7}
water_supply_encoding = {'CORPORATION': 0, 'CORP_BORE': 1, 'BOREWELL': 2}
building_type_encoding = {'IF': 0, 'IH': 1, 'AP': 2, 'GC': 3}

# Home Section
def home():
    st.markdown("# :house: Real Estate Rent Prediction")

    st.markdown("## :red_circle: **Problem Statement:**")
    st.markdown("In the real estate industry, determining the appropriate rental price for a property is crucial for property owners, tenants, and property management companies. Accurate rent predictions can help landlords set competitive prices, tenants make informed rental decisions, and property management companies optimize their portfolio management.")

    st.markdown("## :red_circle: **Goal:**")
    st.markdown("The goal of this project is to develop a data-driven model that predicts the rental price of residential properties based on relevant features. By analyzing historical rental data and property attributes, the model aims to provide accurate and reliable rent predictions.")

    st.markdown("## :red_circle: **Expected Outcomes:**")
    st.markdown("The successful implementation of the house rent prediction model will provide property owners, tenants, and property management companies with a tool to estimate rental prices accurately. This will enhance transparency, aid in decision-making, and contribute to a more efficient and equitable rental market.")

# About Project Section
def about_project():
    st.markdown("# :red[Smart Predictive Modeling for Rental Property Prices]")
    st.markdown("### :red[Technologies Used:] Python,Jupyter notebook,Pickel, Pandas, Numpy, Scikit-Learn, Streamlit, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, ")

    st.markdown('### :red[Overview:]')
    st.markdown(
        'The core objective of this initiative is to design and implement a robust predictive modeling system tailored for estimating rental property prices in the prominent cities of Bangalore and Chennai. This project aims to provide invaluable insights to potential renters, landlords, and property investors by analyzing historical rental data and harnessing the power of advanced machine learning algorithms.')

    st.markdown('### Importance:')
    st.markdown('Rental price of a property is  influenced by a various number of factors:')
    st.markdown(
        '- **Location:** Specific neighborhoods or regions within location can significantly impact rental rates, influenced by amenities, infrastructure, and demand.')
    st.markdown('- **Lease Type & Size:** Variations in lease types such as FAMILY, BACHELOR, and COMPANY.')
    st.markdown(
        '- **Amenities & Facilities:** The presence and quality of amenities like swimming pool,security,AC,Park parking spaces, water supply, gym, and lift.')
    st.markdown(
        'Given the intricate interplay of these influencing elements, the development of a predictive model offers stakeholders a comprehensive, data-driven perspective on rental pricing trends, fostering more informed and strategic decision-making processes.')
    st.markdown("### :red[Domain :] Real Estate & Property Management")

# Predictions Section
def predictions():
    st.markdown("# :red[Predicting Results based on XG Boost Trained Model]")

    st.markdown("### :red[Predicting Rent Price for a property] ")
    # Split the layout into two columns
    col1, col2 = st.columns(2)

    # Input Section - Left Column
    with col1:
        activation_date = st.date_input("Activation Date", datetime.today())
        day, month, year = activation_date.day, activation_date.month, activation_date.year
        property_type = st.selectbox("Type", ['BHK2', 'BHK3', 'BHK1', 'RK1', 'BHK4', 'BHK4PLUS'])
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.000001)
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.000001)
        lease_type = st.selectbox("Lease Type", ['FAMILY', 'ANYONE', 'BACHELOR', 'COMPANY'])
        gym = st.radio("Gym", [0, 1])
        lift = st.radio("Lift", [0, 1])
        swimming_pool = st.radio("Swimming Pool", [0, 1])
        negotiable = st.radio("Negotiable", [0, 1])
        furnishing = st.selectbox("Furnishing", ['NOT_FURNISHED', 'SEMI_FURNISHED', 'FULLY_FURNISHED'])
        parking = st.selectbox("Parking", ['TWO_WHEELER', 'NONE', 'FOUR_WHEELER', 'BOTH'])
        property_size = st.number_input("Property Size", min_value=0)
        property_age = st.number_input("Property Age", min_value=0)
        bathroom = st.number_input("Bathroom", min_value=0)
        facing = st.selectbox("Facing", ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'])
        cup_board = st.number_input("Cupboard", min_value=0)

    # Additional Amenities - Right Column
    with col2:
        floor = st.number_input("Floor", min_value=0)
        total_floor = st.number_input("Total Floor", min_value=0)
        water_supply = st.selectbox("Water Supply", ['CORP_BORE', 'CORPORATION', 'BOREWELL'])
        building_type = st.selectbox("Building Type", ['AP', 'IF', 'IH', 'GC'])
        balconies = st.number_input("Balconies", min_value=0)
        st.subheader("Additional Amenities")
        internet = st.radio("Internet", [0, 1])
        ac = st.radio("AC", [0, 1])
        club = st.radio("Club", [0, 1])
        intercom = st.radio("Intercom", [0, 1])
        servant = st.radio("Servant Room", [0, 1])
        security = st.radio("Security", [0, 1])
        park = st.radio("Park", [0, 1])
        house_keeping = st.radio("Housekeeping", [0, 1])

    # Submit Button
    if st.button("Submit"):
        # Create a DataFrame with the input values
        df_test = pd.DataFrame({
            'type': [type_encoding[property_type]],
            'latitude': [latitude],
            'longitude': [longitude],
            'lease_type': [lease_type_encoding[lease_type]],
            'gym': [gym],
            'lift': [lift],
            'swimming_pool': [swimming_pool],
            'negotiable': [negotiable],
            'furnishing': [furnishing_encoding[furnishing]],
            'parking': [parking_encoding[parking]],
            'property_size': [property_size],
            'property_age': [property_age],
            'bathroom': [bathroom],
            'facing': [facing_encoding[facing]],
            'cup_board': [cup_board],
            'floor': [floor],
            'total_floor': [total_floor],
            'water_supply': [water_supply_encoding[water_supply]],
            'building_type': [building_type_encoding[building_type]],
            'balconies': [balconies],
            'INTERNET': [internet],
            'AC': [ac],
            'CLUB': [club],
            'INTERCOM': [intercom],
            'SERVANT': [servant],
            'SECURITY': [security],
            'PARK': [park],
            'HOUSE_KEEPING': [house_keeping],
            'day': [day],
            'month': [month],
            'year': [year]
        })


        # Display the DataFrame
        st.subheader("DataFrame with Input Values")
        st.write(df_test)

        # Load the scaler
        with open("D:\\GUVI\\project\\rent_prediction\\minmax_scaler_1.pkl", 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        # List of numerical features used for scaling during training
        numerical_features_training = ['latitude', 'longitude', 'property_size', 'property_age', 'bathroom',
                                       'cup_board', 'floor', 'total_floor', 'balconies', 'rent', 'day', 'month', 'year']

        # Extract numerical features from the testing data
        numerical_features_testing = [col for col in df_test.columns if col in numerical_features_training]

        # Ensure the order of features in the testing data matches the order during training
        numerical_features_testing = [feature for feature in numerical_features_training if
                                      feature in numerical_features_testing]

        # Apply the scaler only to the numerical features in the testing data
        df_test[numerical_features_testing] = scaler.transform(df_test[numerical_features_testing])


        # Load the XG Boost model
        with open("D:\\GUVI\\project\\rent_prediction\\xgboost_model.pkl", 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Make predictions
        predictions = loaded_model.predict(df_test)

        # Display the predictions
        st.subheader("Predictions")
        st.write(predictions)

# Sidebar Navigation
def main():
    st.set_page_config(layout="wide", page_title="Rent Prediction App")

    selected = st.sidebar.selectbox("Menu", ["Home", "About Project", "Predictions"])

    if selected == "Home":
        home()

    elif selected == "About Project":
        about_project()

    elif selected == "Predictions":
        predictions()

if __name__ == "__main__":
    main()
