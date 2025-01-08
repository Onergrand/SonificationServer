import base64
import os
import cv2
from ctypes import CDLL, POINTER, c_double
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify

IMAGE_SIZE = 416
N = 176
M = 64
FNAME = 'hificode.wav'
IMAGE_PATH = 'static/blackframe.png'
c_file = './newlib.so'
c_fun = CDLL(c_file)

app = Flask(__name__)


def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def convert_image(frame, contrast):
    gray = cv2.resize(frame, [N, M])
    avg = gray.mean()
    contrast_frame = gray + contrast * (gray - avg)
    contrast_frame = np.clip(contrast_frame, 0, 255)
    type_frame = np.where(contrast_frame == 0, contrast_frame, 10 ** ((contrast_frame / 16 - 15) / 10))
    type_frame = np.flip(np.fliplr(type_frame))
    return type_frame.astype(np.float64)


def save_image(array, output_path):
    newarr = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = Image.fromarray(newarr)
    image.save(output_path)


def merge_save_image_halves(array1, array2, output_path):
    newarr1 = cv2.normalize(array1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    newarr2 = cv2.normalize(array2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    image1 = Image.fromarray(newarr1)
    image2 = Image.fromarray(newarr2)

    if image1.height != image2.height:
        raise ValueError("Высоты изображений не совпадают. Убедитесь, что изображения одинаковой высоты.")

    left_half = image1.crop((0, 0, image1.width // 2, image1.height))
    right_half = image2.crop((image2.width // 2, 0, image2.width, image2.height))

    result_image = Image.new("RGB", (left_half.width + right_half.width, left_half.height))
    result_image.paste(left_half, (0, 0))
    result_image.paste(right_half, (left_half.width, 0))
    result_image.save(output_path)


def encode_media_to_base64(image_path, audio_path):
    try:
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        with open(audio_path, 'rb') as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

        response = {
            'image': encoded_image,
            'audio': encoded_audio,
            'image_type': 'image/png',
            'audio_type': 'audio/wav'
        }
        return response
    except Exception as e:
        return {"error": str(e)}


@app.route('/remake', methods=['POST'])
def remake():
    data = request.json
    image_base64 = data.get('image_base64')
    mode = int(data.get('mode'))
    contrast = float(data.get('contrast'))

    if not image_base64:
        return jsonify({"error": "No image data provided"}), 400

    try:
        image_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    if frame is None:
        return jsonify({"error": "Can't load the image from the provided base64"}), 400

    frame = cv2.resize(frame, (N, M))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_image, second_image = np.split(gray_frame, 2, axis=1)

    try:
        if mode == 0:
            mean_frame = (first_image + second_image) / 2
            new_frame = convert_image(mean_frame, contrast)
            c_fun.stereo(new_frame.ctypes.data_as(POINTER(c_double)))
            save_image(new_frame, IMAGE_PATH)

        elif mode == 1:
            new_first_image = convert_image(first_image, contrast)
            new_second_image = convert_image(second_image, contrast)
            merge_save_image_halves(first_image, second_image, IMAGE_PATH)
            c_fun.binaural(new_first_image.ctypes.data_as(POINTER(c_double)),
                           new_second_image.ctypes.data_as(POINTER(c_double)))

        elif mode == 2:
            newleft = np.empty((M, 13))
            newright = np.empty((M, 13))
            first_image = np.flip(first_image)
            newleft[:, 0] = first_image[:, 0]
            newright[:, 0] = second_image[:, 0]
            fs = 1
            step = 2
            for i in range(1, 13):
                newleft[:, i] = np.mean(first_image[:, fs:fs + step], axis=1)
                newright[:, i] = np.mean(second_image[:, fs:fs + step], axis=1)
                fs += step
                step += 1

            new_first_image = convert_image(newleft, contrast)
            new_second_image = convert_image(newright, contrast)
            merge_save_image_halves(first_image, second_image, IMAGE_PATH)
            c_fun.binaural(new_first_image.ctypes.data_as(POINTER(c_double)),
                           new_second_image.ctypes.data_as(POINTER(c_double)))
        else:
            return jsonify({"status": "fail"})

        response = encode_media_to_base64(IMAGE_PATH, FNAME)
        if "error" in response:
            return jsonify(response), 500

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host='0.0.0.0', port=5000)
