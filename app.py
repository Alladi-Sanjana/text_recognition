import argparse
import json
from typing import Tuple, List
import os

import cv2
import editdistance
from pathlib import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor

from flask import Flask, request, jsonify

app = Flask(__name__)

class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '/content/drive/MyDrive/SimpleHTR/model/charList.txt'
    fn_summary = '/content/drive/MyDrive/SimpleHTR/model/summary.json'
    fn_corpus = '/content/drive/MyDrive/SimpleHTR/data/corpus.txt'

def get_img_height() -> int:
    """Fixed height for NN."""
    return 32

def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())

def infer(model: Model, fn_img: Path) -> str:
    """Recognizes text in image provided by file path."""
    img = cv2.imread(str(fn_img), cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, _ = model.infer_batch(batch, True)
    return recognized[0]

@app.route('/recognize-text', methods=['POST'])
def recognize_text():
    if 'image' not in request.files:
        return "No image file uploaded", 400

    image_file = request.files['image']
    image_path = '/path/to/save/image.jpg'  # Update with the path to save the uploaded image
    image_file.save(image_path)

    model = Model(char_list_from_file(), DecoderType.BestPath, must_restore=True, dump=False)
    recognized_text = infer(model, Path(image_path))

    # Remove the saved image file
    os.remove(image_path)

    return jsonify({'text': recognized_text})

if __name__ == '__main__':
    app.run(debug=True)
