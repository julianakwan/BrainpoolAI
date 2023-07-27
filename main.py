import numpy as np
import matplotlib.pyplot as plt
from openvino.runtime import Core
from PIL import Image
import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#        print('Hello World\n')
        output_image = transform_image(filename)
        output_filename = 'test.jpg'
        output_image.save('static/uploads/'+output_filename)

        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', input_filename=filename, output_filename=output_filename)
    else:
       	flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())


def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

def transform_image(image_name):
    model_path = "/Users/jkwan/Documents/jobs/BrainpoolAI/public/fast-neural-style-mosaic-onnx/fast-neural-style-mosaic-onnx.onnx"
    core = Core()
    core.set_property({'CACHE_DIR': '.../cache'})
    #load the model
    model = core.read_model(model_path)
    #compile the model
    compiled_model = core.compile_model(model=model, device_name="AUTO")
    #AUTO means choose the best processor CPU or GPU
    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)
    network_input_shape = list(input_key.shape)
    network_image_height, network_image_width = network_input_shape[2:]

#    image_file = "/Users/jkwan/Documents/jobs/BrainpoolAI/coco_bike.jpg"
    image_file = "/Users/jkwan/Documents/jobs/BrainpoolAI/static/uploads/"+image_name
    input_image = Image.open(image_file)
#    plt.imshow(input_image)
#    plt.show()

    #resize the image for the model dimensions
    input_image = input_image.resize((network_image_height, network_image_width))
    input_image = np.array(input_image).astype('float32')

    #model expects BRG not RBG
    input_image = np.transpose(input_image,[2,0,1])
    input_image = np.expand_dims(input_image, axis=0)

    result = compiled_model([input_image])[output_key]

    #reverse the pre-processing steps
    result = np.array(result).astype('float32')
    result = result.reshape(3,network_image_height, network_image_width)
    result = result.transpose(1,2,0)

    output_image = Image.fromarray(result.astype("uint8"), 'RGB')
    
    return output_image    
#could resize image again    
#    output_image.resize(
#    plt.imshow(output_image)
#    plt.show()




if __name__ == '__main__':
    app.run()

    
    
