from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('rg_model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index2.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)
    op=(output*100)
    
    if output<0.5:
         return render_template('index2.html', prediction_text='your probability of getting admission is low')
    else:
          return render_template('index2.html', prediction_text='your probability of getting admission is high')
    
 
    
 
        

    
 


if __name__ == '__main__':
    app.run(debug=True)
