# Placement-prediction-
Placement Prediction Website – A web app using machine learning to predict student placement chances with interactive dashboard and analytics.

Placement and Salary Prediction Website

Overview

This project is a web-based application designed to help students and job seekers with placement and salary prediction. It integrates multiple features such as:

Placement Prediction Bot: Predicts placement chances based on input data.

GitHub Analysis: Evaluates GitHub profiles to assess coding and project strength.

Resume Analysis: Provides resume evaluation and improvement suggestions.

Dashboard: Visual representation of predictions and user data.

About Page: Information about the platform.


Project Structure

PLACEMENT AND SALARY PRE...
├── Dataset and training code
│   ├── college_student_placement_dataset.csv
│   ├── placement_dataset.csv
│   ├── package.py
│   ├── placement.py
├── models
│   ├── placement_model_tf.h5
│   ├── salary_model_tf.h5
│   ├── salary_scaler.pkl
│   ├── scalar.pkl
├── static
│   ├── css
│   │   ├── github.css
│   │   ├── mystyle.css
│   │   ├── resume.css
│   │   └── style.css
│   ├── image
│   │   ├── git1.png
│   │   ├── resume.jpg
│   └── js
│       ├── aboutus.jsx
│       ├── dashboard.js
│       ├── github.js
│       ├── main.js
│       └── resume.js
├── templates
│   ├── about.html
│   ├── chatbot.html
│   ├── dashboard.html
│   ├── github_index.html
│   ├── github_results.html
│   ├── index.html
│   ├── resume_index.html
│   └── resume_results.html
├── uploads
├── app.py
├── main.py
├── predictions.json
├── start_app.bat

Features

1. Placement Prediction Bot

Uses machine learning models to predict placement chances and expected salary.



2. GitHub Analysis

Analyzes GitHub repository data for activity, contributions, and coding skills.



3. Resume Analysis

Processes uploaded resumes and provides scoring and suggestions.



4. Dashboard

Displays visual insights using charts and tables.



5. Data and Models

Training datasets and pre-trained models included for predictions.




Technologies Used

Frontend: HTML, CSS, JavaScript, JSX

Backend: Python (Flask/Django assumed)

Machine Learning: TensorFlow/Keras (.h5 models), Scikit-learn (.pkl files)

Data Storage: JSON, CSV

Other Tools: GitHub API integration


How to Run

1. Clone the repository.
2. Ensure Python 3.x is installed.
3. Run the application:
python start_app.bat
4. Open the browser at http://localhost:5000.



Future Improvements

Add more resume parsing features (skills extraction, ATS score).

Expand GitHub analysis (language proficiency, project quality metrics).

Improve UI/UX and add authentication.

Deploy the application online.


