# Disaster Response Pipeline Project
### Project Summary 
The project implements a machine learning pipeline allowing to claasify messages from natural disasters in 36 different categories. This classifier is intended to help disaster reponse organizations to filter relavant messages for the kind of aid service they could provide.

Includes a web app where an emergency worker can input a new message and get classification results in each categoriy and displays three visualizations of the training data.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.
    - Install libraries
        `pip install -r requirements.txt`

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
