from flask import Flask, request, render_template
from pathlib import Path
import os
from feature_extraction import encode_article, SentenceTransformer
import pickle
from data_handler import read_file

app = Flask(__name__)

backbone = None
dont_use_headlines = None

def run_flask(_backbone, _dont_use_headlines):
    global backbone, dont_use_headlines
    backbone = _backbone
    dont_use_headlines = _dont_use_headlines
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    selected_file = request.files['selected-file']
    files_dir = Path(os.path.dirname(__file__)) / "files"
    file_path = files_dir / selected_file.filename
    os.makedirs(files_dir, exist_ok=True)
    selected_file.save(str(file_path))

    backbone_to_use = backbone #'all-mpnet-base-v2'
    model = SentenceTransformer(backbone_to_use)

    file_content = read_file(file_path)
    x = encode_article([file_content], model, encode_headlines_separately=not dont_use_headlines)

    model_path = Path(os.path.dirname(__file__)) / 'model.pkl'
    model = pickle.load(open(model_path, 'rb'))

    y_pred = model.predict(x)

    prediction = "satire"
    if y_pred == 0:
        prediction = "fake"

    return render_template('index.html', prediction=prediction)
