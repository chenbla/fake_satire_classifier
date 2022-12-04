import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def encode_article(article, model, encode_headlines_separately=True):
    """
        Encodes an article into a vector.
        Arguments:
            article: An article to encode.
                     An articles is assumes to be a long sting, with '\n' representing break lines.
                     The first line is the headline, the rest are the body.
            model: A backbone model to use for encoding sentence encoding.
            encode_headlines_separately: If set to True, then the headline and the body are encoded separately,
                                         the article's vector representation is their concatenation. Else, the
                                         headline and the body are flatten into a single sentence and encoded together.
        Returns:
            A vector representation of the article
        """
    if encode_headlines_separately:
        [headline, text] = article.split('\n', 1)
        return np.concatenate((model.encode(headline), model.encode(text)))
    else:
        return model.encode(article)


def feature_extractor(data_dict, encode_headlines_separately=True, backbone_to_use='all-mpnet-base-v2'):
    """
        Encodes the dataset into vectors using pre-trained s-bert model (no FT).

        Arguments:
            data_dict: A dictionary, data_dict['fake'] and data_dict['satire'] stores all the fake and satire articles,
                       respectivly.
            encode_headlines_separately: If set to True, then the headline and the body are encoded separately,
                                         the article's vector representation is their concatenation. Else, the
                                         headline and the body are flatten into a single sentence and encoded together.
            backbone_to_use: The name of the backbone to use for sentences encoding, can be taken from here:
                            https://www.sbert.net/docs/pretrained_models.html

        Returns:
            vecs - a matrix, each column is vector representation for each article
            labels - a vector with the labels of the articles
    """
    model = SentenceTransformer(backbone_to_use)

    data_dict_all = data_dict['fake']+data_dict['satire']
    labels = np.concatenate((np.zeros(len(data_dict['fake'])), np.ones(len(data_dict['satire']))))

    vecs = []
    for article in tqdm(data_dict_all, desc='Encoding articles using pre-trained s-bert model (%s)' % backbone_to_use):
        vecs.append(encode_article(article, model, encode_headlines_separately))
    vecs = np.array(vecs)
    return vecs, labels