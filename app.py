import numpy as np
from flask import Flask, request, render_template
import pickle
import math

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
model_jan = pickle.load(open('model_jan.pkl', 'rb'))
model_feb = pickle.load(open('model_feb.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    del int_features[0]
        
    final_features = [np.array(int_features)]
    if request.form.get("month") == "1":
       prediction = model_jan.predict(final_features)
    elif request.form.get("month") == "2":
       prediction = model_feb.predict(final_features)
    
    
    
    
    count = 0
    
    for i in range(4):
        prediction[0,i] = math.trunc(prediction[0,i])
        if prediction[0,i] <0:
            count = count+1
    
    if count != 0:
        for i in range(4):
            prediction[0,i] = 0
        
    return render_template('index.html', poha='POHA: {}'.format(prediction[0,0]), dosa='DOSA: {}'.format(prediction[0,1]),tea='TEA: {}'.format(prediction[0,2]),coffee='COFFEE :{}'.format(prediction[0,3]))


if __name__ == "__main__":
    app.run(debug=True)
    
    