

import numpy as np
from flask import Flask, render_template
import pickle
from sklearn.preprocessing import StandardScaler
app=Flask(__name__)

model, normalization=pickle.load(open('models/model.pkl','rb'))


@app.route('/')
def home():
	in_data= np.array([1,1,1])
	out_data=pre_processing(in_data.reshape(1,-1),normalization)
	prediction=model.predict(out_data)
	return render_template('index.html',prediction_text=prediction)


@app.route('/predict', methods=['POST'])
def predict():
	in_data= np.array([1,1,1])
	out_data=pre_processing(in_data.reshape(1,-1),normalization)
	prediction=model.predict(out_data)
	return render_template('index.html',prediction_text=prediction)



def pre_processing(in_data, normalization):
	scaler=StandardScaler()
	scaler.mean_=normalization[0]
	scaler.scale_=normalization[1]

	out_data=scaler.transform(in_data)

	return out_data




if __name__=="__main__":
	app.run()

