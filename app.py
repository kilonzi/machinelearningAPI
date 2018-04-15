from flask_api import FlaskAPI
import flask
from flask import Flask, request, json, jsonify
import datetime
import turicreate as tc
import pandas as pd
import requests
import flask_uploads

app = FlaskAPI(__name__)

@app.route('/identify',methods=['POST'])
def identify():
	return "Hello World"

if __name__ == '__main__':
    app.run()
