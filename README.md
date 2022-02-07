# Disaster Response Classifier

### Introduction:

Objective:
  We would like to categorize real messages that were sent during disaster events so that we can send the messages to an appropriate disaster relief agency.

  origin data:![image](picture or gif url)

Overview:
  We will read the dataset provided by Appen, clean the data, and then store it in a SQLite database. We will then create a machine learning pipeline that uses NTLK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model to predict classifications for 36 categories (multi-output classifications). Finally, there is a web app included where an emergency worker can input a new message and get classification results in several categories.

### How to run:

1. Requirements

Python 3
Libraries: numpy, pandas, sqalchemy, re, NLTK, pickle, Sklearn, plotly and flask libraries

2. File descriptions

- app
| - template
| |- master.html  		　    # main page of web app
| |- go.html  			　      # classification result page of web app
|- run.py  			　          # Flask file that runs app

- data
|- disaster_categories.csv　# data to process
|- disaster_messages.csv  　# data to process
|- process_data.py          # Read, clean, and store data
|- InsertDatabaseName.db    # database to save clean data to

- models
|- train_classifier.py      # machine learning pipeline
|- classifier.pkl  　　　　　# saved model

- README.md

3. Instructions

  1.	Data cleaning
  Run the following command in the root directory of the project.
    'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv'
  2.	Model training
  Run the following command in the root directory of the project.
    'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
  3.	Starting the web app
  Go to the directory of the app and run the following command.
    'python run.py'
  4.	Go to http://0.0.0.0:3001/

### Results

1. Message Classification
![image](picture or gif url)
![image](picture or gif url)

2. Model Accuracy

### Acknowledgement
Thanks to Appen for the data set and Udacity for the training.
