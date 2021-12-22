import streamlit as st
import pickle
import pandas as pd
from PIL import Image


im = Image.open("employee_churn.jfif")
st.image(im, width=700)

model = pickle.load(open("final_rf_model", "rb"))
scale = pickle.load(open("mm_scaler.pkl", "rb"))
employee = pd.read_csv('HR_comma_sep.csv')

st.sidebar.header("Select the Employee Features")

satisfaction_level = st.sidebar.slider("Please select the satisfaction level of employee", 1, 100, 50, 5) / 100
last_evaluation = st.sidebar.slider("Please select the last evaluation level of employee", 1, 100, 50, 5) / 100
average_monthly_hour = st.sidebar.slider("Please select the number of hours in average an employee work in a week",
                                         10, 80, 40, 1) * 4
time_spend_company = st.sidebar.slider("Please select the number of years an employee work in company", 1, 20, 5, 1)
project_number = st.sidebar.slider("Please select the number of projects assigned to employee", 0, 15, 5, 1)
productivity = project_number / time_spend_company
work_accident = st.sidebar.selectbox("Please select whether an employee had an accident or not", ['No', 'Yes'])
salary_list = ['low', 'medium', 'high']
salary = st.sidebar.selectbox("Please select salary level of employee", salary_list)

df = pd.DataFrame(data=[[satisfaction_level, last_evaluation, productivity, average_monthly_hour, time_spend_company]],
                  columns=['satisfaction_level', 'last_evaluation', 'productivity', 'average_monthly_hour',
                           'time_spend_company'])

scaled = scale.transform(df)
df = pd.DataFrame(scaled, columns=df.columns)

if salary == 'low':
    df[['salary']] = 0
elif salary == 'medium':
    df[['salary']] = 1
else:
    df[['salary']] = 2

if work_accident == 'No':
    df[['work_accident']] = 0
else:
    df[['work_accident']] = 1

st.header('Single Customer Churn Prediction based on Selected Features')
def single_customer():
    df_table = df
    st.write('')
    st.dataframe(data=df_table, width=1000, height=300)
    st.write('')


single_customer()
st.subheader('Please press **Submit** if everything seems okay')
if st.button("Submit"):
    import time

    with st.spinner("Random Forest Model is loading for Prediction..."):
        my_bar = st.progress(0)
        for p in range(0, 101, 10):
            my_bar.progress(p)
            time.sleep(0.1)

    churn_probability = model.predict_proba(df)

    st.dataframe(churn_probability)

    st.success(f'The Probability of the Employee Churn is %{round(churn_probability[0][1] * 100, 1)}')

    if round(churn_probability[0][1] * 100, 1) > 50:
        st.warning("The Employee is **LEFT**")
    else:
        st.success("The Employee is **STAY**")

st.header('Evaluation of Employees')
employee.drop(["left", 'Department', 'promotion_last_5years'], axis=1, inplace=True)
employee['productivity'] = employee['number_project'] / employee['time_spend_company']
employee.drop(["number_project"], axis=1, inplace=True)
scal = scale.transform(
    employee[['satisfaction_level', 'last_evaluation', 'average_montly_hours', 'time_spend_company', 'productivity']])
df_employee = pd.DataFrame(scal, columns=['satisfaction_level', 'last_evaluation', 'average_montly_hours',
                                          'time_spend_company', 'productivity'])
df_employee = df_employee.join(employee[['Work_accident', 'salary']])
df_employee["salary"] = df_employee["salary"].replace({'low': 1, 'medium': 2, 'high': 3})
st.dataframe(df_employee.head())

pred_probability = [i[1] for i in model.predict_proba(df_employee)]
df_employee["pred_pro"] = pred_probability

if st.checkbox("Select Top Loyal Employees"):
    n = st.slider("Please enter the number of employees to see:", 1, 30, 5, 1)
    st.dataframe(df_employee.sort_values(by="pred_pro", ascending=False).head(n))

elif st.checkbox("Select Employees with Highest Churn Probability"):
    nn = st.slider("Please enter the number of employees to see:", 1, 30, 5, 1)
    st.dataframe(df_employee.sort_values(by="pred_pro").head(nn))
