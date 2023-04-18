from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image

import requests
import grpc
# from grpc.beta import implementations

# Import prediction service functions from TF-Serving API
from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2, get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from utils import label_map_util
from utils import visualization_utils as viz_utils
from core.standard_fields import DetectionResultFields as dt_fields

sys.path.append("..")
tf.get_logger().setLevel('ERROR')

PATH_TO_LABELS = "./data/mscoco_label_map.pbtxt"
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['CLASSIFIED_FOLDER'] = 'static/classified_images/'
app.config['NON_CLASSIFIED_FOLDER'] = 'static/non_classified_images/'

app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def get_stub(host='course_work_tensorflow_serving_container_1', port='8500'):
    channel = grpc.insecure_channel('course_work_tensorflow_serving_container_1:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub

def load_image_into_numpy_array(image):
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def load_input_tensor(input_image):
    image_np = load_image_into_numpy_array(input_image)
    image_np_expanded = np.expand_dims(image_np, axis=0).astype(np.uint8)
    tensor = tf.make_tensor_proto(image_np_expanded)
    return tensor

def inference(frame, stub, model_name='tensorflow_serving_container'):
    
    # Add the RPC command here
    # Call tensorflow server
    # channel = grpc.insecure_channel('localhost:8500')
    channel = grpc.insecure_channel('course_work_tensorflow_serving_container_1:8500', options=(('grpc.enable_http_proxy',0),))
    print("Channel: ", channel)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    print('Stub: ', stub)
    request = predict_pb2.PredictRequest()
    print('Request: ', request)
    request.model_spec.name = 'tensorflow_serving_container'
    
    # cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    input_tensor = load_input_tensor(image)
    request.inputs['input_tensor'].CopyFrom(input_tensor)

    result = stub.Predict(request, 60.0)

    image_np = load_image_into_numpy_array(image)

    output_dict = {}
    output_dict['detection_classes'] = np.squeeze(
        result.outputs[dt_fields.detection_classes].float_val).astype(np.uint8)
    output_dict['detection_boxes'] = np.reshape(
        result.outputs[dt_fields.detection_boxes].float_val, (-1, 4))
    output_dict['detection_scores'] = np.squeeze(
        result.outputs[dt_fields.detection_scores].float_val)

    num_detections = output_dict['detection_boxes'].shape[0]
    
    accuracy_threshold = 0.7
    detected_result = {}

    objects_detected = 0
    class_and_associated_probabilities= []
    for i in range(num_detections):
        if output_dict['detection_scores'][i] >= accuracy_threshold:
            objects_detected = objects_detected + 1
            class_and_associated_probabilities.append({
                'object': 'Object ' + str(i),
                'class': category_index[output_dict['detection_classes'][i]]['name'], 
                'accuracy': output_dict['detection_scores'][i]
                })
    
    detected_result['objects_detected'] = objects_detected
    detected_result['class_and_associated_probabilities'] = class_and_associated_probabilities

    frame = viz_utils.visualize_boxes_and_labels_on_image_array(image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.70,
                agnostic_mode=False)
    return frame, detected_result

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['NON_CLASSIFIED_FOLDER'], filename))
        return uploaded_file(filename=filename)

def uploaded_file(filename):
    PATH_TO_TEST_IMAGES_DIR = app.config['NON_CLASSIFIED_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    IMAGE_SIZE = (12, 8)
    
    stub = get_stub()

    for image_path in TEST_IMAGE_PATHS:
        image_np = np.array(Image.open(image_path))
        image_np_inferenced, detected_result = inference(image_np,stub)
        im = Image.fromarray(image_np_inferenced)
        im.save('static/classified_images/' + filename)
    # print("test images paths", TEST_IMAGE_PATHS)
    non_classified_image_path = os.path.join(app.config['NON_CLASSIFIED_FOLDER'], filename)
    classified_image_path = os.path.join(app.config['CLASSIFIED_FOLDER'], filename)
    detected_result['image_name'] = filename
    detected_result['non_classified_image'] = non_classified_image_path
    detected_result['classified_image'] = classified_image_path
    # print("type(detected_result)", type(detected_result))
    firestore_obj = convert_dict_to_firestore(detected_result)
    # image_files = glob(os.path.join(image_path, "*.jpg"))
    response = store_data(firestore_obj)
    if response.status_code == 404:
        "ss"
    # print("image_files", image_files)
    return redirect(url_for('results', image_path=classified_image_path))

@app.route('/results')
def results():
    # a=1
    # if a==1:
    #     return redirect(url_for('index'))
    # Receive the arguments from the index function
    # firestore_obj = request.args.get('firestore_obj')
    # if response.status_code == 404:
    #     "ss"
    all_results, response = retrieve_data()
    print("all_results all_results", all_results)
    # my_json = json.dumps(all_results)

    if response.status_code == 404:
        "ss"
    # image_path = request.args.get('image_path')  
    root_path = f'http://localhost:5000/'
  
    # Pass the arguments to the results.html
    return render_template('results.html', all_results=all_results, root_path =root_path )



################# Firebase ####################


project_id = "object-detection-6267e"
api_key = "AIzaSyCTiwrPXYvUe2pawcEuWm626omExmpgaK4"
firebase_base_url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents"


collection_name = "results"
firebase_url = f"{firebase_base_url}/{collection_name}?key={api_key}"

##### Store data #####
def store_data(firestore_obj):
    # print("firestore_obj 111111", firestore_obj)
    # print("type(firestore_obj)", type(firestore_obj))
    # firestore_obj = json.loads(firestore_obj)

    # 
    response = requests.post(firebase_url, json=firestore_obj)
    return response

##### Retrieve data #####
def retrieve_data():
    response = requests.get(firebase_url)
    firestore_data = response.json()
    # Convert Firestore documents to array
    
    all_results = convert_firestore_to_dict(firestore_data)

    return all_results, response


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

def convert_firestore_to_dict(firestore_data):
    # print("convert_firestore_to_dictconvert_firestore_to_dict", firestore_data)
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
