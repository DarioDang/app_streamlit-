import streamlit as st
import joblib
import pandas as pd 
import numpy as np
import statistics as stat 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 


try:
    #Read the stroke data
    data_stroke = pd.read_csv("data/stroke_data.csv")
    #Read the diabetes data
    data_diabete = pd.read_csv("data/diabetes_data.csv")
    #Read the heart_attack data
    data_heart_attack = pd.read_csv("data/heart_attack_data.csv")
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# stroke columns selection
data_stroke = data_stroke[['age', 'gender', 'hypertension', 'heart_disease', 'ever_married', 
                    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status','stroke']]

#diabetes columns selection
data_diabete = data_diabete[['Age','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Outcome']]

# heart_attack columns selection
data_heart_attack = data_heart_attack[['Age','Sex','Cholesterol','Heart Rate','Diabetes','Smoking','Obesity','Alcohol Consumption','Diet','Heart Attack Risk']]

# Fill NaN values in the "bmi" column with the mean of the "bmi" column
data_stroke['bmi'].fillna(data_stroke['bmi'].mean(), inplace=True)

#Function to plot the histogram graph
def plot_data_histogram(data,column,title):
    "Write a function to plot the histogram plot"
    plt.figure(figsize=(12, 6))
    axes = plt.axes()
    axes.hist(data[column].dropna(), bins = 30, color = 'skyblue')
    axes.set_title(title)
    axes.set_xlabel(column)
    axes.set_ylabel("Frequency")
    axes.grid(True)
    st.pyplot(plt.gcf()) 

def plot_age_by_condition(data, age_column, condition_column, title):
    """Plot histograms of ages grouped by a condition (e.g., having had a stroke)."""
    plt.figure(figsize=(12, 6))
    # Data for individuals with the condition (assuming '1' means condition is present)
    ages_with_condition = data[data[condition_column] == 1][age_column].dropna()
    # Plotting both histograms on the same axes for comparison
    plt.hist(ages_with_condition, bins=30, alpha=0.5, color='red', label=f'Ages usually get {condition_column}')
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()  # Add a legend to help distinguish the histograms
    st.pyplot(plt.gcf())  # Display the plot in Streamlit

def calculate_average_metrics(data, columns):
    """Calculate the average for specified columns in a DataFrame."""
    averages = {col: data[col].mean() for col in columns}
    return averages

def get_stroke_averages(data_stroke):
    """Get average metrics specifically for stroke data."""
    columns = ['age', 'avg_glucose_level', 'bmi']
    return calculate_average_metrics(data_stroke, columns)

def get_diabetes_averages(data_diabete):
    """Get average metrics specifically for diabetes data."""
    columns = ['Age', 'BloodPressure', 'BMI']
    return calculate_average_metrics(data_diabete, columns)

def get_heart_attack_averages(data_heart_attack):
    """Get average metrics specifically for heart attack data."""
    columns = ['Age', 'Cholesterol', 'Heart Rate']
    return calculate_average_metrics(data_heart_attack, columns)

#Function to compare user to the average data
def plot_user_comparison(user_data, average_data):
    """Plot comparison of user metrics against average metrics."""
    categories = list(user_data.keys())
    user_vals = list(user_data.values())
    average_vals = list(average_data.values())
    fig, ax = plt.subplots()
    ax.plot(categories, user_vals, label="Your Metrics", marker='o')
    ax.plot(categories, average_vals, label="Average Metrics", linestyle="--", marker='o')
    ax.set_ylabel("Metric Values")
    ax.set_title("Your Health Metrics Compare Average Metrics")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


#Using label encoding on categorical variables in 3 datasets.
def label_encode_dataset(data):
    "Write a function to check the categorical variables  using label enconding each categorical in the datasets"
    label_encoder = LabelEncoder()
    # Check if there are any categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        print("Label encoding the following columns:", categorical_columns)
        # Label encode each categorical column
        for column in categorical_columns:
            data[column] = label_encoder.fit_transform(data[column])
    else:
        print("No categorical variables found in the dataset.")

# Label encode each dataset
label_encode_dataset(data_stroke)
label_encode_dataset(data_diabete)
label_encode_dataset(data_heart_attack)

#Function to split the data  
def split_data(data):
    "Function to split the data into training and testing"
    X = data.iloc[:, :-1]  # Exclude the last column as features
    y = data.iloc[:, -1]   # Select the last column as target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Function to train the model for each dataset
def perform_model(X_train, X_test, y_train, y_test):
    "Function to perform the Random Forest classification model"
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, model

#Save the model 
def save_model(model, name):
    "Function to save a model to a file"
    filename = f"{name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")
    
def train_models_and_save(datasets):
    "Function to train the model for each dataset"
    for name, data in datasets:
        print("Random Forest for", name)
        X_train, X_test, y_train, y_test = split_data(data)
        accuracy, model = perform_model(X_train, X_test, y_train, y_test)
        print("Accuracy:", accuracy)
        save_model(model, name)
        print()

# Call the function to train the model for each dataset and save them
    train_models_and_save(datasets)

# Load the saved models for each prediction type 
stroke_model = joblib.load("model/stroke_data_model.joblib")
diabetes_model = joblib.load("model/diabetes_data_model.joblib")
heart_attack_model = joblib.load("model/heart_attack_data_model.joblib")

#Set the function for handling the missing values when the users leave blank
def handle_missing_values(features, data):
    """ Function to handle blank inputs by replacing them with the statistical mode (categorical) or mean (numerical) of each column. """
    features = np.array(features)  # Ensure features is a NumPy array for better handling
    for i in range(features.shape[1]):  # Iterate over each column index
        if pd.isnull(features[:, i]).any():  # Check if there is any NaN in the column
            if data.iloc[:, i].dtype == 'object':
                # Handling categorical data with mode
                # Use try-except to handle cases where the mode might not be straightforward to compute
                try:
                    mode_value = stat.mode(data.iloc[:, i].dropna())
                except stat.StatisticsError:
                    # In case there's no unique mode, we could default to any reasonable value like the first one
                    mode_value = data.iloc[:, i].dropna().iloc[0]
                features[np.where(pd.isnull(features[:, i]))] = mode_value
            else:
                # Handling numerical data with mean
                mean_value = data.iloc[:, i].mean()
                features[np.where(pd.isnull(features[:, i]))] = mean_value
    return features

# Define function to make stroke prediction
def predict_stroke(features):
    "Write a function to make a prediction of stroke"
    try:
        features = handle_missing_values(features, data_stroke)
        prediction = stroke_model.predict(features)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None 

# Define function to make diabetes prediction
def predict_diabetes(features):
    "Write a function to make a prediction of diabetes"
    try:
        features = handle_missing_values(features, data_diabete)
        prediction = diabetes_model.predict(features)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None 

# Define function to make heart_attack prediction
def predict_heart_attack(features):
    "Write a function to make a prediction of heart_attacks"
    try:
        features = handle_missing_values(features, data_heart_attack)
        prediction = heart_attack_model.predict(features)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None 

#Set the wide layout 
st.set_page_config(layout="wide")

# The main function 
def main():
    "Write the main function "
    st.sidebar.title("Our Service") 

    # Define navigation options
    options = ["Home", "Stroke Prediction", "Diabetes Prediction", "Heart Attack Prediction"]

    # Create a sidebar radio button for navigation
    choice = st.sidebar.selectbox("Please choose what you concern", options)

    # Reset all the field when the selection changes
    if "last_choice" not in st.session_state:
        st.session_state.last_choice = choice 
    if choice != st.session_state.last_choice:
        st.session_state.show_bmi = False
        st.session_state.show_glucose = False
        st.session_state.show_age = False
        st.session_state.show_insulin = False
        st.session_state.show_blood_pressure = False
        st.session_state.show_cholesterol = False
        st.session_state.last_choice = choice 

    if choice == 'Home':
        st.sidebar.write("This our home page")
        show_home()
    
    elif choice == "Stroke Prediction":
        st.sidebar.write("Welcome to Stroke Prediction")
        stroke_prediction()

    elif choice == "Diabetes Prediction":
        st.sidebar.write("Welcome to Diabetes Prediction")
        diabetes_prediction()

    elif choice == "Heart Attack Prediction":
        st.sidebar.write("Welcome to Heart Attack Prediction")
        heart_attack_prediction()

# Homepage layout 
def show_home():
    "Write a homepage interface"
    col1, col2 = st.columns([2,1])
    with col1:
        st.title("Welcome To Health Prediction Application")
        st.subheader("Empowering you with insights into your health risks!")
    with col2:
        image_path = 'image/homepage/Homepage.png'
        st.image(image_path)
    
    # Use columns to layout text and images side by side
    col1, col2 = st.columns([1, 2])
    with col1:
        image_path_1 = 'image/homepage/About_us.png'
        st.image(image_path_1)
        
    with col2:
        st.subheader("About this Application")
        st.markdown("""
                    - This application provides **health risk assessments** for stroke, diabetes, and heart attack. 
                    It's designed to give you preliminary information based on general health data.
                    - Please note that the predictions are based on historical data and may not be accurate.
                    Always consult a healthcare professional for medical advice and diagnosis.
                    """)
    
    st.markdown("#### Why Check Your Health Risks?")
    st.markdown("""
                 - **Early Detection**: Identifying risks early can lead to better prevention and management strategies.
                 - **Awareness**: Understanding your health risks can motivate lifestyle changes.
                 -  **Preventive Measures**: Learn ways to reduce your risk through lifestyle and medical interventions.
                 """)
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("How It Works")
        st.markdown("""
                - Simply navigate to the prediction section of your choice using the sidebar. 
                - You'll be prompted to enter relevant health parameters.
                - The system will assess your risk based on the provided information.
                """)
    with col2: 
        image_path = 'image/homepage/How_work.png'
        st.image(image_path)

    col1 , col2 = st.columns([1,2])
    with col1:
        image_path = 'image/homepage/hear_us.png'
        st.image(image_path)
    with col2:
        st.subheader("How did you hear about us?")
        with st.form(key = 'user_info_form', clear_on_submit = True):
            st.selectbox("How did you hear about us?", ["Through a friend", "Online search", "Social media", "Advertisement", "Other"], index = None)
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.write("Thanks you for choosing us!")
    
    # Including some interactive elements
    st.subheader("Engage with Our Interactive Health Tools")
    st.markdown("""
                - **BMI Calculator**: Understand how your body mass index correlates with general health risks.
                - **Diet Recommendations**: Get suggestions based on your health conditions.
                """)
    col1, col2 = st.columns([2, 1])
    #Toogle button for BMI Calculator
    if 'toggle_bmi' not in st.session_state:
        st.session_state['toggle_bmi'] = False
    with col1:
        if st.button("Calculate My BMI" if not st.session_state.toggle_bmi else "Calculate My BMI"):
            st.session_state.toggle_bmi = not st.session_state.toggle_bmi
            if st.session_state.toggle_bmi:
                st.write("BMI Calculator coming soon!")
            else:
                st.write("")
                st.session_state['toggle_bmi'] = False

    # Toggle button for Diet Recommendations
    if 'toggle_diet' not in st.session_state:
        st.session_state['toggle_diet'] = False

    with col2:
        if st.button("Diet Recommendations" if not st.session_state.toggle_diet else "Diet Recommendations"):
            st.session_state.toggle_diet = not st.session_state.toggle_diet
        if st.session_state.toggle_diet:
            st.write("Diet recommendations feature coming soon!")
        else:
            st.write("")
            st.session_state['toggle_diet'] = False

#Keep update the user input into appropriate text file"
def write_to_file(file_path,data):
    "Write user input to a file"
    try:
        file = open(file_path, 'a')
        file.write(','.join(map(str, data)) + '\n')
        file.close()
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")

#Load the user input to predict the function
def read_from_file(file_path):
    "Read the user input data from a file"
    try:
        file = open(file_path, 'r')
        data = file.readline().strip().split(',')
        file.close()
        return data
    except FileNotFoundError:
        print(f" The file {file_path} does not exist.")
        return None 

## Stoke prediction page 
def stroke_prediction():
    "Write the stroke prediction page"
    col1 , col2 = st.columns([2,1])
    with col1:
        st.title("Stroke Prediction")
        st.markdown("""
                A stroke, sometimes called a brain attack, occurs when something blocks blood supply to part of the brain 
                or when a blood vessel in the brain bursts.In either case, parts of the brain become damaged or die. 
                A stroke can cause lasting brain damage, long-term disability, or even death.
                """)
    with col2:
        image_path = 'image/prediction_page/stroke_predict.png'
        st.image(image_path)
    
    st.markdown("Find out more at: [Centers Disease Control](https://www.cdc.gov/stroke/)")

    #Show distribution of BMI across the data
    col1,col2,col3 = st.columns([1,1,1])

    #Initialize session state variables if not already set
    if 'show_bmi' not in st.session_state:
        st.session_state['show_bmi'] = False
    if 'show_glucose' not in st.session_state:
        st.session_state['show_glucose'] = False
    if 'show_age' not in st.session_state:
        st.session_state['show_age'] = False

    with col1:
        if st.button("BMI Distribution"):
            st.session_state.show_bmi = not st.session_state.show_bmi
    with col2:
        if st.button("Glucose Distribution"):
            st.session_state.show_glucose = not st.session_state.show_glucose
    with col3:
        if st.button("Age Distribution"):
            st.session_state.show_age = not st.session_state.show_age
    
    #Display plot below buttons
    if st.session_state.show_bmi:
        plot_data_histogram(data_stroke, 'bmi', 'Distribution of BMI')
    else:
        st.write("")
    if st.session_state.show_glucose:
        plot_data_histogram(data_stroke, 'avg_glucose_level', 'Distribution of Glucose Level')
    else:
        st.write("")
    if st.session_state.show_age:
        plot_age_by_condition(data_stroke, 'age', 'stroke','Common Age That Usually Get Stroke')
    else:
        st.write("")
    
    #Set the predict session
    if 'prediction_made' not in st.session_state:
        st.session_state['prediction_made'] = False 

    # Input the fields for Stroke Prediction
    with st.form(key='heart_attack_prediction_form', clear_on_submit = True):
        age = st.number_input("Age(*)", min_value=20, max_value=150, value= None,placeholder = "Please Enter Your Age")
        gender = st.selectbox("Gender(*)", ["Male", "Female"], index = None, placeholder = "Select Your Gender")
        hypertension = st.selectbox("Hypertension(*) [What is Hypertension?](https://www.who.int/health-topics/hypertension#tab=tab_1)", ["No", "Yes"], index = None, placeholder = "Have you ever have any hypertension?")
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"], index = None, placeholder = "Did you ever had any heart disease?")
        ever_married =  st.selectbox("Ever Married", ["No", "Yes"], index = None, placeholder = "Are you married?")
        work_type = st.selectbox("Work Type", ["Private", "Self_Employed", "Govt_job", "Children", "Never_worked"], index = None)
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], index = None)
        avg_glucose_level = st.number_input("Average Glucose Level(*)", min_value=0.0, max_value=1000.0, value = None, format="%.2f", placeholder = "Standard level is 70 - 126 mg/dL")
        bmi = st.number_input("Body Max Index(*) (BMI) [Find out more BMI](https://en.wikipedia.org/wiki/Body_mass_index)", min_value=0.0, max_value=1000.0, value = None, format="%.1f", placeholder = 'Tell me your BMI')
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "unknown"], index = None)

        # Create a button group for predict and reset buttons
        col1, col2,col3 = st.columns([1,1,1])
        with col1:
            predict_button = st.form_submit_button("Predict Stroke")
        with col2:
            plot_button = st.form_submit_button("Learn More")
        with col3:
            reset_button = st.form_submit_button("Clear Result")
        
        # Perform Stroke Prediction when user clicks the button
        if predict_button:
            if age is None:
                st.error("Age field cannot be blank")
            elif gender is None:
                st.error("Gender field cannot be blank")
            elif hypertension is None:
                st.error("Hypertension field cannot be blank")
            elif avg_glucose_level is None:
                st.error("Average Glucose Level field cannot be blank")
            elif bmi is None:
                st.error("Body Max Index (BMI) cannot be blank")
            else:
                # Encode the data 
                gender_mapping = {"Male": 1, "Female": 0}
                hypertension_mapping = {"No": 0, "Yes": 1}
                heart_disease_mapping = {"No": 0, "Yes": 1}
                ever_married_mapping = {"No": 0, "Yes": 1}
                work_type_mapping = {"Private": 2, "Self_Employed": 3, "Govt_job": 0, "Children": 4, "Never_worked": 1}
                residence_type_mapping = {"Urban": 1, "Rural": 0}
                smoking_status_mapping = {"unknown": 0,"never smoked": 2, "formerly smoked": 1, "smokes": 3}

                # Mapping categorical features
                gender  = gender_mapping[gender] if gender is not None else None
                hypertension  = hypertension_mapping[hypertension] if hypertension is not None else None
                heart_disease = heart_disease_mapping[heart_disease] if heart_disease is not None else None
                ever_married  = ever_married_mapping[ever_married] if ever_married is not None else None
                work_type = work_type_mapping[work_type] if work_type is not None else None
                residence_type = residence_type_mapping[residence_type] if residence_type is not None else None
                smoking_status = smoking_status_mapping[smoking_status] if smoking_status is not None else None

                #Collecting input data
                input_data = [age, gender, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]

                input_data_formatted = map(str,input_data)

                #Specifying file path
                file_path = 'user_record/stroke_user_input_data.txt'

                #Writing to and reading from file
                write_to_file(file_path, input_data_formatted)
                file_data = read_from_file(file_path)

                # Converting data back to required formats 
                file_data = [int(file_data[0]), int(file_data[1]), int(file_data[2]), int(file_data[3]), int(file_data[4]), 
                             int(file_data[5]), int(file_data[6]), float(file_data[7]), float(file_data[8]), int(file_data[9])]

                file_data = [file_data]

                # Call a function to perform prediction using the trained model
                prediction = predict_stroke(file_data)
                st.session_state['prediction_made'] = True
                st.session_state['user_data'] = {'age': age, 'avg_glucose_level': avg_glucose_level, 'bmi': bmi}

                # Display prediction result
                if prediction == 1:
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.error('🚨 High Risk of Stroke 🚨')
                        st.markdown("### Recommended Actions for Stroke Prevention:")
                        st.markdown("""
                            - **Consult a neurologist**: Early professional assessment can significantly manage risks.
                            - **Monitor glucose level regularly**: High average glucose level is a leading cause of stroke.
                            - **Adopt a healthy lifestyle**: Focus on balanced nutrition, regular exercise, and smoking cessation.
                                    """)
                        st.markdown("[Learn more about stroke prevention and care](https://www.stroke.org/en/about-stroke)")
                    with col2:
                        image_path = "image/result/positive_predict.png"
                        st.image(image_path)
                else:
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.success('🟢 Low Risk of Stroke 🟢') 
                        st.markdown("Congratulation! Your results indicate a low risk of stroke. Continue maintaining a healthy lifestyle.")   
                    with col2:
                        image_path = "image/result/negative_predict.png"
                        st.image(image_path)

    #Call the comparision plot function
    if plot_button:
        if st.session_state['prediction_made']:
            user_data = st.session_state['user_data']   
            plot_user_comparison(user_data, get_stroke_averages(data_stroke))
        else:
            st.error("You must predict first before viewing the health metrics comparison.")
    
    # Clear the result 
    if reset_button:
            age = None
            gender = None
            hypertension = None
            heart_disease = None
            ever_married = None
            work_type = None
            residence_type = None
            avg_glucose_level = None
            bmi = None
            smoking_status = None
            st.session_state['prediction_made']  = False
            st.session_state.show_bmi = False
            st.session_state.show_glucose = False
            st.session_state.show_age = False
            st.rerun()

## Diabetes prediction page 
def diabetes_prediction():
    "Write diabetes prediction page"
    col1,col2 = st.columns([2,1])
    with col1:
        st.title("Diabetes Prediction")
        st.markdown("""
                Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin 
                or when the body cannot effectively use the insulin it produces.
                Insulin is a hormone that regulates blood glucose.
                """)
    with col2:
        image_path = 'image/prediction_page/diabetes_predict.png'
        st.image(image_path)

    st.markdown("Find out more at: [World Health Organization](https://www.who.int/news-room/fact-sheets/detail/diabetes)")

    #Show distribution of BMI across the data
    col1, col2, col3 = st.columns([1,1,1])

    #Initalizing session state variables if not already get
    if "show_blood_pressure" not in st.session_state:
        st.session_state['show_blood_pressure'] = False
    if "show_insulin" not in st.session_state:
        st.session_state['show_insulin'] = False 
    if "show_age" not in st.session_state:
        st.session_state['show_age'] = False 

    # Adding to the button
    with col1:
        if st.button("Blood Distribution"):
            st.session_state.show_blood_pressure = not st.session_state.show_blood_pressure
    with col2:
        if st.button("Insulin Distribution"):
            st.session_state.show_insulin = not st.session_state.show_insulin
    with col3:
        if st.button("Age Get Diabetes"):
            st.session_state.show_age = not st.session_state.show_age 
    
    #Display plot below buttons
    if st.session_state.show_blood_pressure:
        plot_data_histogram(data_diabete, 'BloodPressure', 'Distribution of Blood Pressure')
    if st.session_state.show_insulin:
        plot_data_histogram(data_diabete, 'Insulin', 'Distribution of Insulin')
    if st.session_state.show_age:
        plot_age_by_condition(data_diabete, 'Age', 'Outcome','Common Age That Usually Get Diabetes')

    #Set the session state 
    if 'prediction_made' not in st.session_state:
        st.session_state['prediction_made'] = False 
    
    # Input fields for Diabetes Prediction 
    with st.form(key='diabetes_prediction_form', clear_on_submit = True):
        age = st.number_input("Age(*)", min_value=20, max_value=150, value = None, placeholder = "Please enter you age")
        glucose = st.number_input("Glucose (*)[Find more about Glucose](https://www.healthline.com/health/glucose)", min_value=0, max_value=1000, value = None, placeholder = 'Standard level is 70 - 126 mg/dL')
        blood_pressure = st.number_input("Blood Pressure", min_value=20, max_value=1000, value = None, placeholder = "Please give me your blood pressure")
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=1000, value = None, placeholder = "How about your skin?")
        insulin = st.number_input("Insulin [Insulin?](https://my.clevelandclinic.org/health/body/22601-insulin)", min_value=0, max_value=1000, value = None, placeholder = 'Find out more about insulin on the link')
        bmi = st.number_input("Body Max Index(*) (BMI) [Find out more BMI](https://en.wikipedia.org/wiki/Body_mass_index)", min_value=0.0, max_value=1000.0, value = None,format="%.1f",placeholder = "Please provide your BMI")

        # Create a button group for predict and reset buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            predict_button = st.form_submit_button("Predict Diabetes")
        with col2:
            plot_button = st.form_submit_button("Learn More")
        with col3:
            reset_button = st.form_submit_button("Clear Result")

        # Perform Diabetes Prediction when user clicks the button
        if predict_button:
            if age is None:
                st.error("Age field cannot be blank")
            elif bmi is None:
                st.error("BMI field cannot be blank")
            elif glucose is None:
                st.error("Glucose field cannot be blank")
            elif blood_pressure is None:
                st.error("Blood Pressure field cannot be blank")
            else: 
                #Collecting input data
                input_data = [age, glucose, blood_pressure, skin_thickness,insulin, bmi]

                #Formating the collected data
                input_data_formatted = map(str,input_data)

                #Specifying file path
                file_path = 'user_record/diabetes_user_input_data.txt'

                #Writing to and reading from file
                write_to_file(file_path, input_data_formatted)
                file_data = read_from_file(file_path)

                # Converting data back to required formats (ensure this is aligned with your needs)
                file_data = [int(file_data[0]), int(file_data[1]), int(file_data[2]), int(file_data[3]), int(file_data[4]), float(file_data[5])]

                file_data = [file_data]
                # Call the function to predict the model
                prediction = predict_diabetes(file_data)
                st.session_state['prediction_made'] = True
                st.session_state['user_data'] = {'age': age, 'blood_pressure': blood_pressure, 'bmi': bmi}
                # Display the result 
                if prediction == 1:
                    #show_icon_based_on_risk(prediction)
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.error('🚨 High Risk of Diabetes 🚨')
                        st.markdown("""
                                    - **Consult an endocrinologist**: Professional guidance is crucial for managing diabetes.
                                    - **Monitor your glucose levels**: Regular monitoring can help manage your condition effectively.
                                    - **Dietary adjustments**: Focus on low-sugar diets and maintain a healthy weight.
                                    """)
                        st.markdown("[Learn more about managing diabetes](https://www.diabetes.org/diabetes)", unsafe_allow_html=True)
                    with col2:
                        image_path = 'image/result/positive_predict.png'
                        st.image(image_path)
                else:
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.success('🟢 Low Risk of Diabetes 🟢')
                        st.markdown("Congratulation! Your results indicate a low risk of stroke. Continue maintaining a healthy lifestyle.")
                    with col2:
                        image_path = 'image/result/negative_predict.png'
                        st.image(image_path)

    #Call the comparision plot function
    if plot_button:
        if st.session_state['prediction_made']:
            user_data = st.session_state['user_data']   
            plot_user_comparison(user_data, get_diabetes_averages(data_diabete))
        else:
            st.error("You must predict first before viewing the health metrics comparison.")
    
     # Reset input fields if reset button is clicked
    if reset_button:
        age = None
        glucose = None
        blood_pressure = None
        skin_thickness = None
        insulin = None
        bmi = None
        st.session_state['prediction_made']  = False 
        st.session_state.show_blood_pressure = False
        st.session_state.show_insulin = False
        st.session_state.show_age = False
        st.rerun()

# Heart attack prediction page 
def heart_attack_prediction():
    "Write a heart attack prediction page"
    col1 , col2 = st.columns([2,1])
    with col1:
        st.title("Heart Prediction")
        st.markdown("""
                A heart attack, also called a myocardial infarction, happens when a part of the heart muscle doesn't get enough blood. 
                The more time that passes without treatment to restore blood flow, the greater the damage to the heart muscle. 
                Coronary artery disease (CAD) is the main cause of heart attack.
                """)
    with col2:
        image_path = 'image/prediction_page/heart_attack.png'
        st.image(image_path)

    st.markdown("Find out more at: [Wikipedia](https://en.wikipedia.org/wiki/Myocardial_infarction)")

    #Initalizing session state variables if not already get
    if "show_cholesterol" not in st.session_state:
        st.session_state['show_cholesterol'] = False 
    if "show_age" not in st.session_state:
        st.session_state['show_age'] = False 

    #Set up column button
    col1, col2, col3 = st.columns(3)

    # Adding to the button
    with col1:
        if st.button("Cholesterol Distribution"):
            st.session_state.show_cholesterol = not st.session_state.show_cholesterol
    with col3:
        if st.button("Age Get Heart Attack"):
            st.session_state.show_age = not st.session_state.show_age

    #Display plot below buttons
    if st.session_state.show_cholesterol:
        plot_data_histogram(data_heart_attack, 'Cholesterol', 'Distribution of Cholesterol')
    if st.session_state.show_age:
        plot_age_by_condition(data_heart_attack, 'Age', 'Heart Attack Risk','Common Age That Usually Get Heart Attack Risk')

    #Setup the session state for making prediction
    if 'prediction_made' not in st.session_state:
        st.session_state['prediction_made'] = False 

    # Input fields for Heart Attack Prediction
    with st.form(key='heart_attack_prediction_form', clear_on_submit = True):
        age = st.number_input("Age(*)", min_value=20, max_value=150, value = None, placeholder = "Please enter you age")  #st.selectbox("Gender", ["Male", "Female"], index = None, placeholder = "Select Your Gender")
        sex = st.selectbox("Gender(*)", ["Male", "Female"], index = None, placeholder = "Select Your Gender")
        cholesterol = st.number_input("Cholesterol(*) [Find out about cholesterol](https://www.heart.org/en/health-topics/cholesterol/about-cholesterol)", min_value = None, max_value=1000,value = None, placeholder = "Reccommended cholesterol < 200")
        heart_rate = st.number_input("Heart Rate(*)", min_value=40, max_value=1000, value = None, placeholder = "Average heart rate is (60-100/min)")
        diabetes = st.selectbox("Diabetes [Find out more](https://www.who.int/news-room/fact-sheets/detail/diabetes)",["Yes", "No"], index = None, placeholder = "Do you get diabetes?")
        smoking = st.selectbox("Smoking", ["Yes", "No"],index = None, placeholder = "Have you ever smoked?")
        obesity = st.selectbox("Obesity", ["Yes", "No"],index = None, placeholder = "Have you get obesity?")
        alcohol_consumption = st.selectbox("Alcohol Consumption",["Yes", "No"],index = None, placeholder = "Are you an alcoholic?")
        diet = st.selectbox("Diet",["Average", "Unhealthy", "Healthy"],index = None, placeholder = "Tell me about your diet?")

        # Create a button group for predict and reset buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            predict_button = st.form_submit_button("Predict Heart Attack")
        with col2:
            plot_button = st.form_submit_button("Learn More")
        with col3:
            reset_button = st.form_submit_button("Clear Result")
        
        # Perform Heart Attack When users clicks the button
        if predict_button:
            # Perform validation checks
            if age is None:
                st.error(" Age field cannot be blank.")
            elif sex is None:
                st.error("Sex field cannot be blank.")
            elif heart_rate is None:
                st.error(" Heart Rate field cannot be blank.")
            elif cholesterol is None:
                st.error("Cholesterol field cannot be blank")
            else:
                # Encode categorical features fit with model
                sex_mapping = {"Male": 1, "Female": 0}
                diabetes_mapping = {"No": 0, "Yes": 1}
                smoking_mapping = {"No": 0, "Yes": 1}
                obesity_mapping = {"No": 0, "Yes": 1}
                alcohol_consumption_mapping = {"No": 0, "Yes": 1}
                diet_mapping = {"Average": 0, "Unhealthy": 2, "Healthy": 1}

                # Mapping categorical features
                sex = sex_mapping[sex]
                diabetes = diabetes_mapping[diabetes] if diabetes is not None else None
                smoking = smoking_mapping[smoking] if smoking is not None else None
                obesity = obesity_mapping[obesity] if obesity is not None else None
                alcohol_consumption = alcohol_consumption_mapping[alcohol_consumption] if alcohol_consumption is not None else None
                diet = diet_mapping[diet] if diet is not None else None

                #Collecting input data
                input_data = [age, sex, cholesterol, heart_rate, diabetes, smoking, obesity, alcohol_consumption, diet]

                # Formating collected input data
                input_data_formatted = map(str,input_data)

                #Specifying file path
                file_path = 'user_record/heart_user_input_data.txt'

                #Writing to and reading from file
                write_to_file(file_path, input_data_formatted)
                file_data = read_from_file(file_path)

                # Converting data back to required formats (ensure this is aligned with your needs)
                file_data = [int(file_data[0]), int(file_data[1]), int(file_data[2]), int(file_data[3]), int(file_data[4]), float(file_data[5]),
                             int(file_data[6]), int(file_data[7]),int(file_data[8])]

                file_data = [file_data]
                # Call the function to predict the model
                prediction = predict_heart_attack(file_data)
                st.session_state['prediction_made'] = True
                st.session_state['user_data'] = {'age': age, 'heart_rate': heart_rate, 'cholesterol': cholesterol}
                # Display the result
                if prediction == 1:
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.error('🚨 High Risk of Heart Attack 🚨')
                        st.markdown("### Take Immediate Action:")
                        st.markdown("""
                            - **Consult a cardiologist**: Early professional assessment and intervention can save lives.
                            - **Lifestyle adjustments**: Consider immediate changes to your diet, exercise, and smoking habits.
                            - **Stress management**: Engage in stress-reducing activities and consider speaking with a mental health professional.
                                    """)
                        st.markdown("[Learn more about heart disease and prevention](https://www.heart.org/en/health-topics/heart-attack)")
                    with col2:
                        image_path = "image/result/positive_predict.png"
                        st.image(image_path)
                else:
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.success('🟢 Low Risk of Heart Attack 🟢')
                        st.markdown("Congratulation! Your results indicate a low risk of heart attack. Continue maintaining a healthy lifestyle.")
                    with col2:
                        image_path = "image/result/negative_predict.png"
                        st.image(image_path)


         #Call the comparision plot function
        if plot_button:
            if st.session_state['prediction_made']:
                user_data = st.session_state['user_data']   
                plot_user_comparison(user_data, get_heart_attack_averages(data_heart_attack))
            else:
                st.error("You must predict first before viewing the health metrics comparison.")
        
         # Reset input fields if reset button is clicked
        if reset_button:
            age = None
            sex = None
            cholesterol = None
            heart_rate = None
            diabetes = None
            smoking = None
            obesity = None
            alcohol_consumption = None
            diet = None
            st.session_state['prediction_made'] = False
            st.session_state.show_cholesterol = False
            st.session_state.show_age = False
            st.rerun()   

if __name__ == "__main__":
    main()
