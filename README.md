# FORECASTING USER STATISTICS

### Virtual Environment
You need to setup a virtual environment following the below instructions first. (for MACOS, you need to use python instead of python3 and venv/Scripts/activate to activate venv on Windows, also you need to have python>=3.7)
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
```
After setting up the virtual environment, then you need to install the package:
```
python3 -m pip install -e .
```

### Dataset
You can reach the dataset via this [link](https://www.kaggle.com/datasets/wolfgangb33r/usercount). After you download the dataset, you need to move **"app.csv"** file to data directory inside the project.


### Serving
After you provide the dataset, you can run the forecast/main.py file to generate the artifacts and it will print the predictions for the next 6 hours using the latest available 24 hours data.

Also, you can interact with the system using fastapi framework after running the forecast/main.py file. Then, you need to use the following command and go to /docs to benefit from fastapi's UI. It will be running on **localhost:8000**.
```
uvicorn app.api:api
```

If you run forecast/main.py, you will get the predictions for the next 6 hours and total number of users for the next 24 hours respectively. In the last line, there will be a simple report on memory usage and execution time. Through fastapi, you can send get requests to get these predictions. Production-ready code is not complete yet, the get responses are created to provide only the results for the challenge.