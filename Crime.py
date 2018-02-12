from flask import Flask, request
from flask import render_template
import re
import numpy as np
import send

#Web App
trained_clf,num_unique_dayofweek,num_unique_pddistrict,dayofweek_dict,pddistrict_dict,category_set = send.rforest();
app = Flask(__name__, static_url_path='/static')
app.static_folder = 'static'
@app.route('/')
def container():
	 return render_template('container.html')

@app.route('/login',methods = ['POST', 'GET'])
def log():
	if request.method == 'POST':
		address = request.form['ad']
		time = request.form['tme']
		pddistrict_string = request.form['di']
		date = request.form['da']
		dayofweek_string = request.form['day']
		longitude = request.form['lo']
		latitude = request.form['la']
		X_test = []
		date = re.search("([0-9]{1,2})/([0-9]{1,2})/([0-9]{4})", date).groups()
	 
		date = [int(x) for x in date]
		
		
		time = re.search("([0-9]{1,2}):([0-9]{2})",time).groups()
		# time is of the form [hour, minute, second]
		time = [int(x) for x in time]
		#pddistrict_string = float(pddistrict_string);
	
		X_row = date + time + [longitude, latitude, \
		dayofweek_string, pddistrict_string]
		X_test.append(X_row);
		# one-hot encoding from existing dicts
		
		for i, row in enumerate(X_test):
			encoded_dayofweek = [0]*num_unique_dayofweek
			encoded_pddistrict = [0]*num_unique_pddistrict
			current_dayofweek = row[-2]
			current_pddistrict = row[-1]
			try:
				encoded_dayofweek[dayofweek_dict[current_dayofweek]] = 1
			except KeyError:
				encoded_dayofweek[0] = 1
			try:
				encoded_pddistrict[pddistrict_dict[current_pddistrict]] = 1
			except KeyError:
				encoded_pddistrict[0] = 1
			X_test[i] = row[:-2] + encoded_dayofweek + encoded_pddistrict
		

			 # write predicted probabilities to file
		work=-1.0;
		#num_classes = trained_clf.best_estimator_.n_classes_
		x = 0;
		percentage = np.empty(10, dtype=int)
		percentage1 = np.empty(10, dtype=float)
		predicted_probas = trained_clf.predict_proba(X_test)
		for i, prediction in enumerate(predicted_probas):
			for j in range(10):
				x+=prediction[j]
			for j in range(10):
				percentage[j]= (prediction[j]/x)*100;
				percentage1[j]=percentage[j]*30;
		#type_of_crime='burglary'
		return render_template("result.html",predict = percentage,predict1 = percentage1) 	

if __name__ == '__main__':
	 app.run(debug = True)