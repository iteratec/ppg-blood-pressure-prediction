# Blood Pressure prediction
This repository allows you to 
-	make blood pressure predictions on datasets recorded from the O2Ring (https://getwellue.com/pages/o2ring-oxygen-monitor). 
-	test different machine learning algorithms on features extracted from the UCI Cuff-Less Blood Pressure Estimation Data Set (https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation#) 

For more information about the method and the extracted features read our article (Link).


## Installing Requirements
To install the requirements use poetry. Poetry is a tool for dependency management and packaging in Python
(https://python-poetry.org/).

- install poetry (https://python-poetry.org/docs/)
    - osx / linux / bashonwindows install instructions: 

    `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`
    - windows powershell install instructions: 

    `(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -`
- open a new shell
- navigate to the project folder
- run `poetry install`



## Run prediction
- open _make_prediction.py_
- modify parameters at top
    - add your height and age
    - choose a dataset from the _Dataset_ folder
    - set `furst_run = True` is this is your first prediction otherwise set it to `False`
- run _make_prediction.py_

## Machine Learning predictions
For the machine learning models we used the UCI Cuff-Less Blood Pressure Estimation Data Set (https://archive.ics.uci.edu/ml/datasets/Cuff-Less+Blood+Pressure+Estimation#)

**Models:**
-	Linear Regression
-	Random Forest
-	Neural Network

**Dataset:**

From the UCI Dataset we extracted nine features and the systolic and diastolic blood pressure values and create our own dataset to train on.
Our dataset is devided into three parts.
- dataset_part1.pkl
- dataset_part2.pkl
- dataset_part3.pkl

Because they contain the wrong blood pressure values there are three parts only containing the systolic and diastolic blood pressure values.
- dataset_bp1.pkl
- dataset_bp2.pkl
- dataset_bp3.pkl


