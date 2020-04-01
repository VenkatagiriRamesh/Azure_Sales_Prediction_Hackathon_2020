from flask import *  
import os
import re
import pickle
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import os
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

model= joblib.load(os.path.join(os.getcwd(), 'vote.pkl'))

input_sample = pd.DataFrame(data=[{'Column3': None, 'Column4': None, 'Column5': 37148.2, 'Column6': 2045.282, 'Column7': None, 'Column8': '', 'Column9': None, 'Column10': None, 'Column11': 7.446882968, 'Column12': None, 'Column13': 100.0, 'Column14': 32.95, 'Column15': 0.515, 'Column16': 8.226, 'Column17': 0.0132580645, 'Column18': 42744.25684, 'Column19': 42.41, 'Column20': 49.09144714, 'Column21': 9.691793869, 'Column22': None, 'Column23': 1.688895637, 'Column24': 1.427532412, 'Column25': 8696587915.0, 'Column26': 39.44102455, 'Column27': 2.611781593, 'Column28': 0.0339, 'Column29': 35.8170301, 'Column30': None, 'Column31': None, 'Column32': 97.17336466, 'Column33': 35.5573706, 'Column34': 44.5027166, 'Column35': 63.37726834, 'Column36': 1728388.673, 'Column37': 331927.5394, 'Column38': 0.1627315206, 'Column39': 40.5605634}])
output_sample = np.array([0])

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})

app = Flask(__name__)

def model_predict(file_path, model_path):

    print(file_path)
    file = file_path.split(".")[1]
    if(file=='csv'):
        df = pd.read_csv(file_path)
    elif(file=='xlsx'):
        df = pd.read_excel(file_path)
    idx = 0  
    n = df.shape[0]
    new_col=[]
    for i in range(0,n):
        new_col.append(0)
    df.insert(loc=idx, column='Year',value=new_col)
    year = []
    period = []
    for index,row in df.iterrows():
        x = str(row['Period'])
        year.append(x[:4])
        p = x[-2:]
        p=re.sub("[^0-9]", "", p)
        period.append(p)

    df['Year']=year
    df['Period']=period
    x=df.drop(columns=['EQ'])
    y_test=df['EQ']
    y_pred=model.predict(x)
    print('Deleting File at Path: ' + file_path)
    os.remove(file_path)
    print('Deleting File at Path - Success - ')
    return y_pred

 
@app.route('/')  
def upload():
	return render_template("main.html")  
 
@app.route('/predict', methods = ['POST'])  
def success():  
    if request.method == 'POST':
        f = request.files['file']  
        f.save(os.path.join("static/uploads",f.filename)) 
        print('Begin Model Prediction...')
        file_path = "static/uploads/"+f.filename
        model_path = "static/model/vote.pkl" 			#Edit model name
        preds = model_predict(file_path, model_path)
        print('End Model Prediction...')
        print(preds)
        return render_template("result.html", resGet=preds) 
  
if __name__ == '__main__':  
    app.run(host="0.0.0.0",port=80,debug = True)  