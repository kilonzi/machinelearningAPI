from flask_api import FlaskAPI
from flask import Flask, request, json, jsonify
import datetime
from datetime import datetime
import turicreate as tc
import pandas as pd
import requests
#import os
from flask import Flask, request, redirect, url_for
#from werkzeug.utils import secure_filename
#import string
#import random
#from random import randint,choice
import flask_uploads
#from flask_uploads import UploadSet, configure_uploads, IMAGES,UploadNotAllowed

#UPLOAD_FOLDER = '/models'
#ALLOWED_EXTENSIONS = set(['txt', 'csv'])

app = FlaskAPI(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 0.0016 * 1024 * 1024
	
#files = UploadSet('files', extensions="csv")

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
    app.run(host='0.0.0.0', port=5000, debug=True)
