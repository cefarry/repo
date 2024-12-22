# import json
# import streamlit as st
# import requests

# user_options = {}

# st.title('White Wine Quality Prediction')

# streamlit_options = json.load(open("streamlit_options.json"))
# # streamlit_options

# for field_name, range in streamlit_options["slider_fields"].items():
#     min_val, max_val = range
#     current_value = round((min_val + max_val)/2)
#     user_options[field_name] = st.sidebar.slider(field_name, min_val, max_val, value = current_value)

# user_options


# if st.button('Predict'):
#     data = json.dumps(user_options, indent=2)
#     r = requests.post('http://127.0.0.1:8000/predict', json=data)
#     st.write(r.json()) 


import json
import streamlit as st
import requests

# Dictionary to store user inputs
user_options = {}

# Title of the app
st.title('White Wine Quality Prediction')

# Load slider configuration
streamlit_options = json.load(open("streamlit_options.json"))

# Create sliders for input fields
for field_name, range_vals in streamlit_options["slider_fields"].items():
    min_val, max_val = range_vals
    current_value = round((min_val + max_val) / 2)
    user_options[field_name] = st.sidebar.slider(
        field_name, min_val, max_val, value=current_value
    )

user_options

# Button to make prediction
if st.button('Predict'):
    # Wrap user_options under "features"
    data = {"features": [user_options]}
    # Send POST request to FastAPI
    response = requests.post('http://127.0.0.1:8000/predict', json=data)
    
    # Check if request was successful
    if response.status_code == 200:
        prediction = response.json().get('predictions', [])[0]  # Extract the prediction
        st.write(f'Predicted Quality: {prediction}')
    else:
        st.write(f"Error: {response.json()}")
