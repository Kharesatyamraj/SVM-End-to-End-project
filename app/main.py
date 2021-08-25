
from flask import Flask,request
import pandas as pd
import numpy as np
import flasgger
from flasgger import Swagger
from sklearn.svm import SVC





app = Flask(__name__)
Swagger(app)
#pickle_in = open ("model_rbf.pkl","rb")
model_rbf = SVC(kernel = "rbf")
model = model_rbf

@app.route('/')
def home():
      return "Welcome to the  prediction page"
@app.route('/predict')
def iris_prediction():
    
    """ Lets find the flower species
    This is using docstrings for specifications.
    ---
    parameters:
      - name      : sepal_length
        in        : query
        type     : number
        required : true
      - name      : sepal_width
        in        : query
        type     : number
        required : true
      - name      : petal_length
        in        : query
        type     : number
        required : true
      - name      : petal_width
        in        : query
        type     : number
        required : true
    responses:
        200:
            description: The output values
    """
    sepal_length  =  request.args.get('sepal_length')
    sepal_width   =  request.args.get('sepal_width')
    petal_length  =  request.args.get('petal_length')
    petal_width   =  request.args.get('petal_width')
    prediction= model.predict([[sepal_length,sepal_width,petal_length,petal_width]]) 
    return "The predicted is" + str(prediction)



if __name__ == '__main__':
     app.run(host='0.0.0.0',port=8000)