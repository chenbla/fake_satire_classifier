from pathlib import Path
import os
from data_handler import read_data
from feature_extraction import feature_extractor
from classify import model_selection
import pickle
import argparse


def offline_model_training(backbone='all-mpnet-base-v2', dont_use_headlines=True):
        # read the data
        print("reading the data")
        dataset_path = Path(os.path.dirname(__file__)) / "FakeNewsData" / "FakeNewsData" / "StoryText 2"
        fake_dataset_dir = dataset_path / "Fake" / "finalFake"
        satire_dataset_dir = dataset_path / "Satire" / "finalSatire"
        data_dict = read_data(fake_dataset_dir, satire_dataset_dir)

        # encode to vectors
        print("performing feature extraction")
        x, y = feature_extractor(data_dict,
                                 backbone_to_use=backbone,
                                 encode_headlines_separately=not dont_use_headlines)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', action='store_true', default=False)
    parser.add_argument('--backbone', default='all-MiniLM-L6-v2')
    parser.add_argument('--dont_use_headlines',  action='store_false')
    args = parser.parse_args()

    backbone = args.backbone
    dont_use_headlines = args.dont_use_headlines

    offline_model_training(backbone=args.backbone, dont_use_headlines=args.dont_use_headlines)
