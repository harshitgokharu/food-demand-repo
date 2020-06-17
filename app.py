import numpy as np
from flask import Flask, request, render_template
import pickle
import math

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
model_jan = pickle.load(open('model_jan.pkl', 'rb'))
model_feb = pickle.load(open('model_feb.pkl', 'rb'))
model_mar = pickle.load(open('model_mar.pkl', 'rb'))
model_apr = pickle.load(open('model_apr.pkl', 'rb'))
model_may = pickle.load(open('model_may.pkl', 'rb'))
model_jun = pickle.load(open('model_jun.pkl', 'rb'))
model_jul = pickle.load(open('model_jul.pkl', 'rb'))
model_aug = pickle.load(open('model_aug.pkl', 'rb'))
model_sep = pickle.load(open('model_sep.pkl', 'rb'))
model_oct = pickle.load(open('model_oct.pkl', 'rb'))
model_nov = pickle.load(open('model_nov.pkl', 'rb'))
model_dec = pickle.load(open('model_dec.pkl', 'rb'))


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
    elif request.form.get("month") == "3":
       prediction = model_mar.predict(final_features)
    elif request.form.get("month") == "4":
       prediction = model_apr.predict(final_features)
    elif request.form.get("month") == "5":
       prediction = model_may.predict(final_features)
    elif request.form.get("month") == "6":
       prediction = model_jun.predict(final_features)
    elif request.form.get("month") == "7":
       prediction = model_jul.predict(final_features)
    elif request.form.get("month") == "8":
       prediction = model_aug.predict(final_features)
    elif request.form.get("month") == "9":
       prediction = model_sep.predict(final_features)
    elif request.form.get("month") == "10":
       prediction = model_oct.predict(final_features)
    elif request.form.get("month") == "11":
       prediction = model_nov.predict(final_features)
    elif request.form.get("month") == "12":
       prediction = model_dec.predict(final_features)
    
    
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
    
    