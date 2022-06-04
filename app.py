
from flask import Flask, render_template, request,jsonify
import pickle
import numpy as np
# filename = 'diabetes-prediction-rfc-model.pkl'
# classifier = pickle.load(open(filename, 'rb'))
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    Age =int(request.form.get('age'))
    Gender =int(request.form.get('gender'))
    Polyuria =int(request.form.get('polyuria'))
    Polydipsia =int(request.form.get('polydipsia'))    
    sudden  =int(request.form.get('sudden_weight_loss'))    
    Irritability =int(request.form.get('irritability'))    
    Itching =int(request.form.get('itching'))
    partial =int(request.form.get('partial_paresis'))    
    Alopecia =int(request.form.get('alopecia'))    
    delayed =int(request.form.get('delayed_healing'))    
    inputquery=np.array([[Age,Gender,Polyuria,Polydipsia,sudden,Itching,Irritability,delayed,partial,Alopecia]]);    
    inputquery=np.array([[40,0,0,0,0,0,0,0,0,0]])
    #print(inputquery)    
    #result = model.predict(inputquery)  
    result = model.predict(inputquery.tolist()).tolist()
    print(result)
    #data = np.array([[1, 1, 1, 10, 25, 52.36, 12.53, 29]])       
    #result= classifier.predict(data)   
    #return jsonify({'you have diabetes':result})        
    #return render_template('sub.html', prediction=result)
    #if result==[1]:
        #return ('you have diabetes-please consult doctor')
    #else:
        #return ('you don`t have diabetes')
    return render_template('sub.html', prediction=result)

if __name__ == '__main__':
	app.run(debug=True)


        
    #return render_template('sub.html', prediction=my_prediction)


#from functools import partial
#from optparse import Values
#from flask import Flask,request,jsonify
#import pickle
#import numpy as np

#model = pickle.load(open('model.pkl','rb'))
#app = Flask(__name__)

#@app.route('/')
#def home():
    #return "hello world"

#@app.route('/predict',methods='POST')
#def predict():
    #Age = request.form.get('Age')
    #Gender = request.form.get('Gender')
    #Polyuria = request.form.get('Polyuria')
    #Polydipsia = request.form.get('Polydipsia')
    #sudden = request.form.get('sudden weight loss')
    #Itching = request.form.get('Itching')
    #Irritability = request.form.get('Irritability')
    #delayed = request.form.get('delayed healing')
    #partial = request.form.get('partial perisis')
    #Alopecia = request.form.get('Alopecia')

    


    #input_query = np.any(np.array(Age,Polyuria,Polydipsia,sudden,Itching,Irritability,delayed,partial,Alopecia))
    #input_query = input_query.reshape((1,-1))
    #input_query = input_query.values.reshape(1,-1)

    #result = model.predict(input_query)0

    #return jsonify({'sugar':((result))})
#if __name__ == '__main__':
    #app.run(debug=True)