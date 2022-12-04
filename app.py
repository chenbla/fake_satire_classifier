import argparse
from flask import Flask, request, render_template
from pathlib import Path
import os
from feature_extraction import encode_article2, AutoModel, AutoTokenizer
import pickle
from data_handler import read_file

app = Flask(__name__)


backbone = 'all-MiniLM-L6-v2'
dont_use_headlines = True


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

    backbone_to_use = backbone
    model = AutoModel.from_pretrained("sentence-transformers/%s" % backbone_to_use)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/%s" % backbone_to_use)

    file_content = read_file(file_path)
    x = encode_article2([file_content], model, tokenizer, encode_headlines_separately=not dont_use_headlines)

    model_path = Path(os.path.dirname(__file__)) / 'model.pkl'
    model = pickle.load(open(model_path, 'rb'))

    y_pred = model.predict([x])

    prediction = "satire"
    if y_pred == 0:
        prediction = "fake"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', action='store_true', default=False)
    parser.add_argument('--backbone', default='all-MiniLM-L6-v2')
    parser.add_argument('--dont_use_headlines',  action='store_false')
    args = parser.parse_args()

    backbone = args.backbone
    dont_use_headlines = args.dont_use_headlines

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)


