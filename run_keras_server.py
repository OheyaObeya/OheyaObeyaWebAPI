import io
import logging
import os
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import flask

from result_formatter import ResultFormatter, ThreeLevelResultFormatter
import mylogger

# curl -X POST -F image=@推論したい画像.jpg 'http://localhost:5000/predict'

logger = logging.getLogger('OheyaObeya')
logger = mylogger.setup(logger, log_level=logging.DEBUG)

app = flask.Flask(__name__)
model = None
# MODEL_NAME = '20190227-234407_mobilenet_aug_oo_best_model.h5'
MODEL_NAME = '20190227-234407_mobilenet_aug_oo_best_model_saved_model'

graph = tf.get_default_graph()


def load_oheyaobeya_model(path: str):  # TODO
    global model

    if Path(path).suffix == '.h5':
        print('???????????????????????????')
        print('???????????????????????????')
        print('???????????????????????????')

        model = load_model(path)
    else:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # SavedModel
        # MobileNetが読み込めないKerasの不具合に対応するための処理
        # https://github.com/tensorflow/tensorflow/issues/22697
        model = tf.contrib.saved_model.load_keras_model(path)
        print(model.summary())


def preprocess(image, img_shape: Tuple[int]) -> np.ndarray:
    # TODO: これはモデルによって変わる grayscaleを期待している場合は、以下のコメントアウトを外す
    # expected_image_mode = 'L'  # grayscale
    # if image.mode != expected_image_mode:
    #     image = image.convert(expected_image_mode)
    # if img_shape[2] == 1:
    #     image = image.convert(expected_image_mode)

    img_size = (img_shape[0], img_shape[1])
    image = image.resize(img_size)
    image = img_to_array(image)
    image = image.astype('float32')
    image /= 255
    logger.debug('image.shape = {}'.format(image.shape))
    # (28, 28, 1) -> (1, 28, 28, 1)
    image = np.expand_dims(image, axis=0)

    return image


def predict_core(image, rformatter: ResultFormatter) -> dict:
    # Preprocess
    # TODO: 起動時にtraining時のjsonを読み込んで実行するようにすると修正の必要がなくなる
    # TODO: ここは、モデルによって指定するサイズが変わる
    img_shape = (128, 128, 3)
    image = preprocess(image, img_shape=img_shape)

    # Predict
    # result sample: array([[0.00529605, 0.99470395]], dtype=float32)
    preds = model.predict(image)
    logger.debug(preds)

    # Convert
    data = rformatter.convert(preds)
    logger.debug(data)

    data['model_version'] = MODEL_NAME

    return data


@app.route('/predict', methods=['POST'])
def predict():
    results = {'success': False}

    if flask.request.method == 'POST':
        if flask.request.files.get('image'):
            image = flask.request.files['image'].read()
            image = Image.open(io.BytesIO(image))

            global graph
            with graph.as_default():
                results = predict_core(image, ThreeLevelResultFormatter())
                results['success'] = True

    return flask.jsonify(results)


if __name__ == '__main__':
    print(('Loading Keras model ...'))
    model_path = Path(__file__).parent / 'model' / MODEL_NAME
    load_oheyaobeya_model(str(model_path))
    app.debug = True
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
