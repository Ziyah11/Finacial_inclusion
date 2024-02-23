import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
encoder = LabelEncoder()
scaler = StandardScaler()


data = pd.read_csv('Financial_inclusion_dataset.csv')
df = data.copy()


st.title("FiNACIAL INCLUSION IN AFRICA ")
# st.markdown("<h1 style = 'color: #1F4172; text-align: center; font-family: helvetica '>STARTUP PREDICTION PROJECT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Ziyah</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html= True)
st.image('pngwing.com (2).png', width = 350, use_column_width = True )

st.markdown("<br>", unsafe_allow_html= True)
st.write(' This is a Machine learning project that predict which individuals are most likely to have or use a bank account by exploring the demorgraphic information and finacial services.')
st.markdown("<br>", unsafe_allow_html = True)
# st.dataframe(data, use_container_width = True )

# st.sidebar.image('pngwing.com (4).png', caption= 'Welcome User')

# st.sidebar.write('Feature Input')


scaler = StandardScaler()

# df.drop(['uniqueid'], axis=1, inplace=True)

encoders = {}

for i in data.select_dtypes(include = 'number').columns:
    df[i] = scaler.fit_transform(df[[i]])

for i in data.select_dtypes(exclude = 'number').columns:
    encoder = LabelEncoder()
    df[i] = encoder.fit_transform(df[i])
    encoders[i + '_encoder'] = encoder

st.markdown("<h4 style='color: #1F4172; text-align: center; font-family: Arial, sans-serif;'>DATA</h4>", unsafe_allow_html=True)
st.dataframe(data)


x = df.drop('bank_account', axis=1)
y = df.bank_account

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, stratify=y)

model = LogisticRegression()
model.fit(xtrain, ytrain)

st.markdown("<h4 style='color: #1F4172; text-align: center; font-family: Arial, sans-serif;'>PREDICTOR MODEL</h4>", unsafe_allow_html=True)
# st.dataframe(data)


st.sidebar.image('pngwing.com (4).png', width=100, use_column_width=True, caption='Welcome User')
st.markdown("<br>", unsafe_allow_html=True)

country = st.sidebar.selectbox('COUNTRY INTERVIEWEE IS IN', data['country'].unique())
year = st.sidebar.number_input('YEAR', data['year'].min(), data['year'].max())
location_type = st.sidebar.selectbox('TYPE OF LOCATION', data['location_type'].unique())
cellphone_access = st.sidebar.selectbox('IF INTERVIEWEE HAS ACCESS TO A CELLPHONE', data.cellphone_access.unique())
household_size = st.sidebar.number_input('HOUSEHOLD SIZE', data['household_size'].min(), data['household_size'].max())
gender_of_respondent = st.sidebar.selectbox('GENDER OF INTERVIEWEE', data.gender_of_respondent.unique())
age_of_respondent = st.sidebar.number_input('RESPONDENT AGE', data['age_of_respondent'].min(), data['age_of_respondent'].max())
relationship_with_head = st.sidebar.selectbox('THE INTERVIEWEE RELATIONSHIP WITH THE HEAD OF THE HOUSE', data.relationship_with_head.unique())
marital_status = st.sidebar.selectbox('THE MARITAL STATUS OF THE INTERVIEWEE', data.marital_status.unique())
educational_level = st.sidebar.selectbox('HIGHEST LEVEL OF EDUCATION', data['education_level'].unique())
job_type = st.sidebar.selectbox('TYPE OF JOB INTERVIEWEE HAS', data.job_type.unique())

new_country = encoders['country_encoder'].transform([country])
new_location_type = encoders['location_type_encoder'].transform([location_type])
new_cellphone_access = encoders['cellphone_access_encoder'].transform([cellphone_access])
new_gender_of_respondent = encoders['gender_of_respondent_encoder'].transform([gender_of_respondent])
new_relationship_with_head = encoders['relationship_with_head_encoder'].transform([relationship_with_head])
new_marital_status = encoders['marital_status_encoder'].transform([marital_status])
new_educational_level = encoders['education_level_encoder'].transform([educational_level])
new_job_type = encoders['job_type_encoder'].transform([job_type])


input_var = pd.DataFrame({
    'country': [new_country],
    'year': [year],
    'location_type': [new_location_type],
    'cellphone_access': [new_cellphone_access],
    'household_size': [household_size],
    'age_of_respondent': [age_of_respondent],
    'gender_of_respondent': [new_gender_of_respondent],
    'relationship_with_head': [new_relationship_with_head],
    'marital_status': [new_marital_status],
    'education_level': [new_educational_level],
    'job_type': [new_job_type]
})



st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h5 style='margin: -30px; color: olive; font:sans-serif' >", unsafe_allow_html=True)
st.dataframe(input_var)

# Check column order and feature names
#print("Input_var columns:", input_var.columns)
#print("Model training columns:", xtrain.columns)



prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred:
        # Include the prediction step here
        predicted = model.predict(input_var)
        output = 'NOT HAVING A BANK ACCOUNT' if predicted[0] == 0 else 'HAVING A BANK ACCOUNT'
        st.success(f'The individual is predicted to {output}')
        st.balloons()


import plotly.express as px
fig = px.pie(names=['Not Having a Bank Account', 'Having a Bank Account'], values=[80, 20], title='Prediction Distribution')
st.plotly_chart(fig)
