import base64
import datetime
from io import BytesIO
import json
from flask import Flask, render_template
from flask import request
import os
from keras.models import load_model
import numpy as np
import pandas as pd
import scipy
import sklearn
import os
from PIL import Image
import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io
import pickle
# display, transform, read, split ...
import numpy as np
import cv2 as cv
import os
import splitfolders
import matplotlib.pyplot as plt

# tensorflow
import tensorflow.keras as keras
import tensorflow as tf

# image processing
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

# model / neural network
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH, 'static/models/')
UPLOAD_FOLDER = os.path.join(BASE_PATH, 'static/image/')

##------------------------------LOAD MODELS -----------------------------
# model_sgd_path = os.path.join(MODEL_PATH,'dsa_image_classification_7_sgd.pickle')
# scaler_path = os.path.join(MODEL_PATH,'dsa_scaler_7.pickle')
# model_sgd = pickle.load(open(model_sgd_path,'rb'))
# scaler = pickle.load(open(scaler_path,'rb'))
model_sgd_path = os.path.join(MODEL_PATH,'resnet50_new.h5')
model = load_model(model_sgd_path)

# uploaded Images folder path
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'image')
# Check if the folder directory exists, if not then create it
if not os.path.exists(app.config['UPLOAD_FOLDER'] ):
    os.makedirs(app.config['UPLOAD_FOLDER'] )


@app.errorhandler(404)
def error404(error):
    message="error404"
    return render_template("error.html",message=message)

@app.errorhandler(405)
def error405(error):
    message="error405"
    return render_template("error.html",message=message)

@app.errorhandler(500)
def error500(error):
    message="error500"
    return render_template("error.html",message=message)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":
        upload_file =request.files['image_name']
        filename=upload_file.filename
        print('The filename has been uploaded =',filename)
        #knows the extensions of the file
        ext = filename.split('.')[-1]
        print('The extention of the filename=',ext)
        if ext.lower() in ['png','jpg','jpeg']:
            path_save = os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print('File saved sucessfully')
            # send to pipeline model
            results = pipeline_model(path_save,model)
            hei = getheight(path_save)
            print(results)
            return render_template('upload.html',fileupload=True,extension=False,data=results,image_filename=filename,height=hei)
        else:
            print('Use only the extention with .jpg, .png, .jpeg')

            return render_template('upload.html',extension=True,fileupload=False)


    else:
        return render_template('upload.html',fileupload=False)

# @app.route('/')
# def about():
#     return render_template("capture.html")

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    filename = ''  # using filename variable to display video feed and captured image alternatively on the same page
    image_data_url = request.form.get('image')

    if request.method == 'POST' and image_data_url:
        try:
            # Decode the base64 data URL to obtain the image data
            image_data = base64.b64decode(image_data_url.split(',')[1])
            
            # Create an image from the decoded data
            img = Image.open(BytesIO(image_data))
            
            # Convert the image to RGB mode if it's in RGBA mode
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Generate a filename with the current date and time
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            upload_file_name = f"leaf_{timestamp}.jpg"
            
            # Save the image to the upload folder
            upload_file_path = os.path.join(UPLOAD_PATH, upload_file_name)
            img.save(upload_file_path, 'JPEG')
            print('File saved successfully')

            # Send to the pipeline model
            results = pipeline_model(upload_file_path,model)
            hei = getheight(upload_file_path)
            print(results)

            # Display the results on the template
            return render_template('capture.html', fileupload=True, extension=False, data=results,image_filename=upload_file_name,height=hei)
            
        except IndexError as e:
            error_message = f'Error processing image: {str(e)}'
            return render_template('capture.html', filename=filename, error_message=error_message)

    return render_template('capture.html', filename=filename)
    

def getheight(path):
    img = skimage.io.imread(path)
    h,w,_ =img.shape
    aspect = h/w
    given_width = 100
    height = given_width*aspect
    return height


      
    
# MODELING A PIPELINE
def pipeline_model(path,model):
    # pipeline model
    # have already loaded the image
    # image = skimage.io.imread(path)
    # RESIZE AND PROCESS IMAGE
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    # CONVERTING IMAGE INTO ARRAY
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.array([img_array])
    # generate predictions for samples
    predictions = model.predict(img_array)
    # define classes name
    class_names = ['Asoca','Changalamparanda','Illipa','Nannari','Thipalli']
    # generate argmax for predictions
    class_id = np.argmax(predictions, axis = 1)
    predictions = predictions.flatten()
    # cal. z score
    z = scipy.stats.zscore(predictions)
    prob_value = scipy.special.softmax(z)
    # getting top three probabilty values
    top_2_prob_ind = prob_value.argsort()[::-1][:3]
    # for making pipeline clasnames should be defined in array
    top_labels = [class_names[class_id.item()] for class_id in top_2_prob_ind]
    top_prob = prob_value[top_2_prob_ind]
    top_dict = dict()
    for key,val in zip(top_labels,top_prob):
    #     top_dict.update({key:np.round(val,3)})
        top_dict.update({key: {"probability": np.round(val, 3), "details": get_plant_details(key)}})
    return top_dict
def get_plant_details(plant_name):
    # Corrected file path
    json_file_path = os.path.join('static', 'js', 'plant.json')
    # Load plant details from JSON file
    with open(json_file_path, 'r') as file:
        plant_details = json.load(file)

    return plant_details.get(plant_name, {"description": "Details not available."})       

if __name__ == "__main__":
    app.run(debug=True)