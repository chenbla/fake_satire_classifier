from pathlib import Path
import os
from data_handler import read_data
from feature_extraction import feature_extractor
from classify import model_selection
import argparse
import pickle

from flask_handler import run_flask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', action='store_true')
    parser.add_argument('--backbone', default='all-mpnet-base-v2')
    parser.add_argument('--dont_use_headlines',  action='store_false')
    args = parser.parse_args()

    if args.train_mode:
        # read the data
        print("reading the data")
        dataset_path = Path(os.path.dirname(__file__)) / "FakeNewsData" / "FakeNewsData" / "StoryText 2"
        fake_dataset_dir = dataset_path / "Fake" / "finalFake"
        satire_dataset_dir = dataset_path / "Satire" / "finalSatire"
        data_dict = read_data(fake_dataset_dir, satire_dataset_dir)

        # encode to vectors sing s-bert
        print("performing feature extraction using s-bert")
        x, y = feature_extractor(data_dict,
                                 backbone_to_use=args.backbone,
                                 encode_headlines_separately=not args.dont_use_headlines)

        # find the best classifier
        print("performing model selection")
        model = model_selection(x, y)

        # train the model with all the data
        print("fitting the selected model")
        model.fit(x, y)

        # save the model
        print("saving the model")
        pickle_out = open("model.pkl", "wb")
        pickle.dump(model, pickle_out)
        pickle_out.close()
    else:
        run_flask()


