from flask_api import FlaskAPI
from flask import Flask, request, json, jsonify
import datetime
import turicreate as tc
import pandas as pd
import requests
import flask_uploads

app = FlaskAPI(__name__)

@app.route('/identify',methods=['POST'])
def identify():
	#Loading of train data
	url = "https://bireum.com/version-test/fileupload/f1523461200006x770887485239654800/x323ScHhf5zrpvBvSHmNwqLW2KW9ezhyosf8QrZ1cmXBZIIcKv.csv"
	sf_train   = tc.SFrame.read_csv(url,error_bad_lines=True)

	#input_post_json
	json_data = request.get_json(force = True)

	#converts the JSON to a dataframe then a S Frame
	df = pd.DataFrame(data = json_data,index=[0])
	sf_new = tc.SFrame(data=df)

	#Gets the maximum number of options we would like to get
	#You will need to assign a category colum
	a = len((sf_train["category"].unique()))

	#Starts the prediction part
	m = tc.nearest_neighbor_classifier.create(sf_train, target='category')
	ystar = m.predict_topk(sf_new, max_neighbors=10, k=a)

	empty_json={}
	result = False
	while result is False:
	    try:
	        for i in range(a):
	            empty_json[ystar["class"][i]]=ystar["probability"][i]
	            result = True
	    except:
	        pass
	return json.dumps(empty_json)

if __name__ == '__main__':
    app.run()
