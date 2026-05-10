import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_FILE = "C:\\Users\\USER\Desktop\student perform\student_performance_dataset.csv"
USER_FILE = "users.csv"

# Ensure dataset exists
if not os.path.exists(DATA_FILE):
    df = pd.DataFrame(columns=['Student_ID','Study_Hours','Attendance',
                               'Previous_Marks','Assignments','Internal_Marks','Final_Result'])
    df.to_csv(DATA_FILE, index=False)

if not os.path.exists(USER_FILE):
    users = pd.DataFrame(columns=['Username','Password'])
    users.to_csv(USER_FILE, index=False)

# Load data
data = pd.read_csv(DATA_FILE)
users = pd.read_csv(USER_FILE)

# Clean dataset to avoid NaN errors
data = data.fillna({
    'Study_Hours': 0,
    'Attendance': 0,
    'Previous_Marks': 0,
    'Assignments': 0,
    'Internal_Marks': 0,
    'Final_Result': "Fail"
})

# Page navigation
if "page" not in st.session_state:
    st.session_state.page = "welcome"

# --- Welcome Page ---
if st.session_state.page == "welcome":
    st.title("🎓 Welcome to Student Performance Prediction System")
    if st.button("Predict", key="predict_button"):
        st.session_state.page = "login"
        st.rerun()

# --- Login / Registration Page ---
elif st.session_state.page == "login":
    st.title("🔐 Login / Register")
    username = st.text_input("Enter Username", key="username_input")
    password = st.text_input("Enter Password", type="password", key="password_input")

    if st.button("Login", key="login_button"):
        if ((users['Username'] == username) & (users['Password'] == password)).any():
            st.success("Login Successful!")
            st.session_state.page = "student"
            st.rerun()
        else:
            st.error("User not found. Please register.")

    if st.button("Register", key="register_button"):
        if ((users['Username'] == username)).any():
            st.error("Username already exists!")
        else:
            new_user = pd.DataFrame([[username,password]], columns=['Username','Password'])
            users = pd.concat([users,new_user], ignore_index=True)
            users.to_csv(USER_FILE, index=False)
            st.success("Registration Successful! Please login.")

# --- Student Page ---
elif st.session_state.page == "student":
    st.title("📊 Student Prediction Portal")

    student_id = st.text_input("Enter Student ID", key="student_id_input")

    if student_id:
        if str(student_id) in data['Student_ID'].astype(str).values:
            # Existing student → predict
            student = data[data['Student_ID'].astype(str)==str(student_id)]
            X = data[['Study_Hours','Attendance','Previous_Marks','Assignments','Internal_Marks']]
            y = data['Final_Result']

            if len(data) > 5:
                X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

                models = {
                    "Logistic Regression": LogisticRegression(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Naive Bayes": GaussianNB(),
                    "Support Vector Machine": SVC()
                }

                results = {}
                for name, model in models.items():
                    model.fit(X_train,y_train)
                    y_pred = model.predict(X_test)
                    results[name] = accuracy_score(y_test, y_pred)

                st.subheader("Model Accuracy Comparison")
                st.write(pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']))

                prediction = models["Logistic Regression"].predict(student[['Study_Hours','Attendance','Previous_Marks','Assignments','Internal_Marks']])[0]
                st.success(f"Prediction for {student_id}: {prediction}")

                # Graphs
                st.subheader("Graphs Showing Trends")
                fig1, ax1 = plt.subplots()
                sns.scatterplot(x=data['Study_Hours'], y=data['Previous_Marks'], hue=data['Final_Result'], palette="viridis", ax=ax1)
                ax1.set_title("Study Hours vs Previous Marks")
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                sns.boxplot(x=data['Final_Result'], y=data['Attendance'], palette="Set2", ax=ax2)
                ax2.set_title("Attendance vs Result")
                st.pyplot(fig2)

                # At-risk students
                st.subheader("List of At-Risk Students")
                at_risk = data[(data['Attendance'] < 70) | (data['Final_Result'] == "Fail")]
                st.write(at_risk[['Student_ID','Attendance','Previous_Marks','Final_Result']])

            else:
                st.warning("Not enough data to train models yet.")

        else:
            st.info("New student detected. Please enter details:")

            study_hours = st.number_input("Study Hours",0,50,5, key="study_hours_input")
            attendance = st.number_input("Attendance (%)",0,100,75, key="attendance_input")
            previous_marks = st.number_input("Previous Marks",0,100,70, key="marks_input")
            assignments = st.number_input("Assignments Score",0,100,80, key="assignments_input")
            internal_marks = st.number_input("Internal Marks",0,100,60, key="internal_input")

            if st.button("Save Student", key="save_button"):
                prediction = "Pass" if previous_marks>=50 and attendance>=50 else "Fail"
                new_row = pd.DataFrame([[str(student_id),study_hours,attendance,
                                         previous_marks,assignments,internal_marks,prediction]],
                                       columns=['Student_ID','Study_Hours','Attendance',
                                                'Previous_Marks','Assignments','Internal_Marks','Final_Result'])
                data = pd.concat([data,new_row], ignore_index=True)
                data.to_csv(DATA_FILE,index=False)
                st.success(f"New student {student_id} added. Prediction: {prediction}")
                st.write("Saved record:")
                st.write(new_row)

                # Show updated ML results and graphs
                if len(data) > 5:
                    X = data[['Study_Hours','Attendance','Previous_Marks','Assignments','Internal_Marks']]
                    y = data['Final_Result']
                    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

                    models = {
                        "Logistic Regression": LogisticRegression(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "Naive Bayes": GaussianNB(),
                        "Support Vector Machine": SVC()
                    }

                    results = {}
                    for name, model in models.items():
                        model.fit(X_train,y_train)
                        y_pred = model.predict(X_test)
                        results[name] = accuracy_score(y_test, y_pred)

                    st.subheader("Model Accuracy Comparison")
                    st.write(pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']))

                    st.subheader("Graphs Showing Trends")
                    fig1, ax1 = plt.subplots()
                    sns.scatterplot(x=data['Study_Hours'], y=data['Previous_Marks'], hue=data['Final_Result'], palette="viridis", ax=ax1)
                    ax1.set_title("Study Hours vs Previous Marks")
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots()
                    sns.boxplot(x=data['Final_Result'], y=data['Attendance'], palette="Set2", ax=ax2)
                    ax2.set_title("Attendance vs Result")
                    st.pyplot(fig2)

                    st.subheader("List of At-Risk Students")
                    at_risk = data[(data['Attendance'] < 70) | (data['Final_Result'] == "Fail")]
                    st.write(at_risk[['Student_ID','Attendance','Previous_Marks','Final_Result']])

                   