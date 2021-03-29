from flask import Flask, jsonify, request
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './images'

length, heigth = 150, 150
model_file = './model/model.h5'  # Se importa el modelo
weights_model = './model/weights.h5'  # Se importa los pesos del modelo
model = load_model(model_file)
model.load_weights(weights_model)
print(">>>>>Model loaded")


@app.route('/')
def home():
    return 'api-pet-prediction ok'


@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['imageFile']
    # Save Image
    filename = secure_filename(image.filename)
    name_image = 'image.' + filename.split('.')[1]
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], name_image))

    # Predict
    predict_result = model_predict('./images/' + name_image)
    # Identify
    pet = identify_pet(predict_result)
    return jsonify({
        "pet": pet,
        "result": predict_result[0].tolist()
    })


def model_predict(file):
    x = load_img(file, target_size=(length, heigth))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    array = model.predict(images, batch_size=20)
    print(array)
    return array


def identify_pet(array):
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        return "Cat"
    elif answer == 1:
        return "Dog"
    elif answer == 2:
        return "Rabbit"


if __name__ == "__main__":
    print(">>>>App run")
    app.run()
