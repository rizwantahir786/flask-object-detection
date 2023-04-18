########################## Import packages or libraries ##########################

import os # To interact with operating system 
import sys # To interact with system specific parameters and functions

import grpc # grpc library for the communication between clients and servers
import numpy as np # To perform mathematical operations for arrays and matrices
import requests # to send http requests using python
import tensorflow as tf #python module for working with tensorflow machine learning framework
from flask import Flask, redirect, render_template, request, url_for # imports from flask for building 
# web application in python and handle https requests and render HTML templates
from PIL import Image # package to open, manipulate and save different image file formats
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
# Above modules provide functions for working with gRPC to make predictions with a
# TensorFlow model hosted on a server
from werkzeug.utils import secure_filename # utility function for securely generating a random filename for uploaded files

from core.standard_fields import DetectionResultFields as dt_fields
# Above package defines a set of standard fields that are used to represent the results of an object detection task, 
# such as the bounding box coordinates, class labels, and confidence scores of detected objects

from utils import label_map_util # utility to work with label maps, which are used in object detection


from utils import visualization_utils as viz_utils
# Above module provide function for the visualization of detected results on images using Matplotlib

#################################################################################

########################## Firebase Rest Api Configuration ##########################
project_id = "object-detection-6267e" # Assigning firebase project id to the variable
api_key = "AIzaSyCTiwrPXYvUe2pawcEuWm626omExmpgaK4" # Assigning firebase api key to the variable
firebase_base_url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents"
# base URL for accessing the results collection within the firebase

collection_name = "results" # collection name of firebase
firebase_url = f"{firebase_base_url}/{collection_name}?key={api_key}"
# complete URL for accessing the results collection within the firebase with firebase api key

####################################################################################

sys.path.append("..") # appends the parent directory of the current working directory to the Python path
tf.get_logger().setLevel('ERROR') # to log the messages with severity level 'ERROR'

PATH_TO_LABELS = "./data/mscoco_label_map.pbtxt" # path to the label map file
NUM_CLASSES = 90 # number of classes that should be predicted

label_map = label_map_util.load_labelmap(PATH_TO_LABELS) # to Load the label map from a file at PATH_TO_LABELS 

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
# Above line of code converts a label map into a list of category dictionaries, where each dictionary contains information 
# about a single class such as its id, name, and display name

category_index = label_map_util.create_category_index(categories) #It creates a category index from the list of category dictionaries

app = Flask(__name__) # Flask library that to create a new Flask application

app.config['CLASSIFIED_FOLDER'] = 'static/classified_images/' # assigning classfied images path to confguration variable
app.config['NON_CLASSIFIED_FOLDER'] = 'static/non_classified_images/' # assigning non-classfied images path to confguration variable

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])  # assigning set of allowed image extensions to the configuration variable


def allowed_file(filename): 
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
# Above function checks whether a given filename has an allowed file extension.

def get_stub(host='tensorflow_serving_container', port='8500'):
    channel = grpc.insecure_channel('tensorflow_serving_container:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub
# Above function is to create a reusable gRPC stub that can be used to make multiple requests to a TensorFlow Serving server. 
# This is used to make predictions with a TensorFlow model.

def load_image_into_numpy_array(image):
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
# Above function converts PIL (Python Imaging Library) Image object into a NumPy array.

def load_input_tensor(input_image):
    image_np = load_image_into_numpy_array(input_image)
    image_np_expanded = np.expand_dims(image_np, axis=0).astype(np.uint8)
    tensor = tf.make_tensor_proto(image_np_expanded)
    return tensor
# Above function takes an input image and returns a TensorFlow tensor that can be used as input to a TensorFlow model.


# Below function performs object detection inference on a given image frame using a TensorFlow Serving
def inference(frame, stub, model_name='tensorflow_serving_container'):
    
    # Establish an insecure gRPC channel to the TensorFlow Serving instance
    channel = grpc.insecure_channel('tensorflow_serving_container:8500', options=(('grpc.enable_http_proxy',0),))
    print("Channel: ", channel)

    # Create a gRPC stub for making predictions
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    print('Stub: ', stub)

    # Create a new prediction request
    request = predict_pb2.PredictRequest()
    print('Request: ', request)

    # Set the model name to be used for prediction
    request.model_spec.name = 'tensorflow_serving_container'
    
    # Load the input image as a numpy array and create a tensor to send to the model for prediction
    image = Image.fromarray(frame)
    input_tensor = load_input_tensor(image)
    request.inputs['input_tensor'].CopyFrom(input_tensor)

    # Make a prediction using the gRPC stub and the input tensor
    result = stub.Predict(request, 60.0)

    # Convert the input image to a numpy array
    image_np = load_image_into_numpy_array(image)

    # Extract the output values from the prediction result and store them in a dictionary
    output_dict = {}
    output_dict['detection_classes'] = np.squeeze(
        result.outputs[dt_fields.detection_classes].float_val).astype(np.uint8)
    output_dict['detection_boxes'] = np.reshape(
        result.outputs[dt_fields.detection_boxes].float_val, (-1, 4))
    output_dict['detection_scores'] = np.squeeze(
        result.outputs[dt_fields.detection_scores].float_val)

    # Get the number of detections from the output dictionary
    num_detections = output_dict['detection_boxes'].shape[0]
    
    # Set an accuracy threshold to filter out detections
    accuracy_threshold = 0.3
    detected_result = {} # To save number of objects, classes detected and their probabilities

    objects_detected = 0 # variable to count total number of objects detected
    class_and_associated_probabilities= [] # to save class and probabilities of objects detected
    for i in range(num_detections): # for loop on the total number of detections with a check of accuracy threshold to store values in above defined variables
        if output_dict['detection_scores'][i] >= accuracy_threshold:
            objects_detected = objects_detected + 1
            class_and_associated_probabilities.append({
                'object': 'Object ' + str(i),
                'class': category_index[output_dict['detection_classes'][i]]['name'], 
                'accuracy': output_dict['detection_scores'][i]
                })
    
    detected_result['objects_detected'] = objects_detected
    detected_result['class_and_associated_probabilities'] = class_and_associated_probabilities

    # Visualize the detected objects and their associated probabilities on the input image
    frame = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np, #input image, in the form of a NumPy array
                output_dict['detection_boxes'], # list of bounding box coordinates for each detected object in the image
                output_dict['detection_classes'], # list of class IDs corresponding to each detected object in the image
                output_dict['detection_scores'], # list of confidence scores corresponding to each detected object in the image
                category_index, # dictionary mapping class IDs to class names
                use_normalized_coordinates=True, # to treat the bounding box coordinates as normalized coordinates
                max_boxes_to_draw=200, # number of bounding boxes to draw on the image
                min_score_thresh=.3, # minimum confidence score threshold required to display a bounding box
                agnostic_mode=False # to avoid displaying all detections as a single class
                )
    return frame, detected_result # returning frame and detected_result

@app.route('/')
def index():
    return render_template('index.html')
# Above function defines a route for the root URL of the app, which is index.html

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['NON_CLASSIFIED_FOLDER'], filename))
        return uploaded_file(filename=filename)
# Above function accepts POST requests for uploading an image file. 
# First it checks if the file is allowed or not. If the file is valid than it will
# save it to the NON_CLASSIFIED_FOLDER directory with a secure filename

def uploaded_file(filename):
    # Define the path to the test image directory
    PATH_TO_TEST_IMAGES_DIR = app.config['NON_CLASSIFIED_FOLDER']
    # Define the path to the test image
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    
    # Get the gRPC stub
    stub = get_stub()
    # for loop over the test images and perform inference on each one
    for image_path in TEST_IMAGE_PATHS:
        # Load the image as a numpy array
        image_np = np.array(Image.open(image_path))
        # Perform inference on the image and get the inferred image and detected result
        image_np_inferenced, detected_result = inference(image_np,stub)
        # Convert the inferred image numpy array back to an Image object
        im = Image.fromarray(image_np_inferenced)
        # Save the inferred image to the classified images directory
        im.save('static/classified_images/' + filename)
    # Define the paths to the non-classified and classified images
    non_classified_image_path = os.path.join(app.config['NON_CLASSIFIED_FOLDER'], filename)
    classified_image_path = os.path.join(app.config['CLASSIFIED_FOLDER'], filename)
    # Add the image name, non-classified image path, and classified image path to the detected result dictionary
    detected_result['image_name'] = filename
    detected_result['non_classified_image'] = non_classified_image_path
    detected_result['classified_image'] = classified_image_path
   
    # Convert the detected result dictionary to a Firestore object
    firestore_obj = convert_dict_to_firestore(detected_result)
    # Store the Firestore object in the database and get any error messages if occurs
    error = store_data(firestore_obj)
    
    # Redirect the user to the classified image details page with the detected result and error message
    return redirect(url_for('classfied_image_details', detected_result= detected_result, error=error ))


@app.route('/classfied_image_details')
def classfied_image_details():
    # Get the detected_result data passed through the URL query parameters
    detected_result = request.args.get('detected_result')
    # root path using the current request's host  
    root_path = "http://" + request.host + "/"
    # Render the classfied_image_details.html template with the detected_result and root_path variables passed to it
    return render_template('classfied_image_details.html', detected_result=detected_result, root_path =root_path )


@app.route('/results')
def results():
    # retrieve all the data from Firestore
    all_results, error = retrieve_data()
    # root path using the current request's host  
    root_path = "http://" + request.host + "/"
    # render the results template and pass the retrieved data and root path as arguments
    return render_template('results.html', all_results=all_results, root_path =root_path, error=error )



################# Firebase Functions####################


##### Store data #####
def store_data(firestore_obj):
    # store the detected result of an image and store it in the firestore database using HTTP POST request
    response = requests.post(firebase_url, json=firestore_obj)
    if response.status_code == 404: # If error than save error message to display at front-side
        error = response.message
    else:
        error=""

    return error # return error message string 

##### Retrieve data #####
def retrieve_data():
    # Get Firestore data
    response = requests.get(firebase_url)
    # Convert Firestore response to JSON
    firestore_data = response.json()
    # Convert Firestore documents to array
    if response.status_code == 404:
        error = response.message
        all_results= []
    elif response.json() == {} :
        error = "No record(s) found!"
        all_results=[]
    else:
        # Convert Firestore documents to array
        all_results = convert_firestore_to_dict(firestore_data)
        error=""

    
    return all_results, error

# Function to convert dictionary to firestore object so it can be save as a document in firebase collection
def convert_dict_to_firestore(detected_result):
    firestore_obj = {'fields': {}}
    for key, value in detected_result.items():
        if isinstance(value, int):
            firestore_obj['fields'][key] = {'integerValue': value}
        elif isinstance(value, str):
            firestore_obj['fields'][key] = {'stringValue': value}
        elif isinstance(value, float):
            firestore_obj['fields'][key] = {'doubleValue': value}
        elif isinstance(value, list):
            array_value = {'values': []}
            for item in value:
                array_value['values'].append({'mapValue': {'fields': convert_dict_to_firestore(item)['fields']}})
            firestore_obj['fields'][key] = {'arrayValue': array_value}
        elif isinstance(value, dict):
            firestore_obj['fields'][key] = {'mapValue': {'fields': convert_dict_to_firestore(value)['fields']}}
    return firestore_obj


# Function to convert firestore object to dictionary so it can be displayed at the front-side

def convert_firestore_to_dict(firestore_data):
    
    output = []
    documents = firestore_data['documents']

    for document in documents:
        result = {}
        fields = document["fields"]
        result["objects_detected"] = int(fields["objects_detected"]["integerValue"])
        result["image_name"] = fields["image_name"]["stringValue"]
        result["non_classified_image"] = fields["non_classified_image"]["stringValue"]
        result["classified_image"] = fields["classified_image"]["stringValue"]
        class_probabilities = fields["class_and_associated_probabilities"]["arrayValue"]["values"]
        class_and_probabilities = []
        for cp in class_probabilities:
            cp_fields = cp["mapValue"]["fields"]
            class_and_probabilities.append({
            "object": cp_fields["class"]["stringValue"],
            "class": cp_fields["class"]["stringValue"],
            "accuracy": cp_fields["accuracy"]["doubleValue"]
            })
        result["class_and_associated_probabilities"] = class_and_probabilities
        output.append(result)


    
    return output

if __name__ == '__main__':
    app.run(debug =True,host='0.0.0.0', port=5000)
