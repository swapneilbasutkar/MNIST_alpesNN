from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import re
import base64
import cv2
import sys 
import os
import requests
import json
from PIL import Image

#API Endpoint 
url = 'http://api.alpes.ai/snn/'
TOKEN = 'Token vGSa5ZUPnU8M3emNmq5C2TTWLHBqQMBZ2nZSsqMq'


app = Flask(__name__)

    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))    
    cv2.imwrite('alpesprocess.png',x)
    im = Image.open('alpesprocess.png')
    tv = list(im.getdata())
    tva = [(x) for x in tv]
    #print(tva)
    str1 = ','.join(str(e) for e in tva)
    #print(str1)
    # Predict
    api_method = 'predict'
    headers = {'authorization': TOKEN}
    params = {'model': 'model_5400776','kvalue': 5,'nbest': 0.3}
    data = {'test_features':str1} 
    r = requests.post(url+api_method,params=params,json=data,headers=headers)
    response=json.loads(r.text)
    #print(type(response))
    print("Predicted label:",response['predected_label'])
    return str(response['predected_label'])

   
    
def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='127.0.0.1', port=port)
