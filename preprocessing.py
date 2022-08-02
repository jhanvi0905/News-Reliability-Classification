from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import re
from wordcloud import WordCloud
import numpy as np
import torch, torchtext
from torchtext.functional import to_tensor
import random


class PreprocessData:

    """"Takes Dataset in csv format, loads, cleans and batches it into model readable format
        
    Parameters
    ----------
    link_to_corpus : str
    stopwords_dict: dictionary of stopwords
    
    Functions 
    -------
    Returns Cleaned DataFrame Object 
    Yields Data Insights
    """""

    def __init__(self, link_to_corpus, stopwords_dict):

        self.stopwords_dict = stopwords_dict
        self.link_to_corpus = link_to_corpus
        self.clean_dataset = None

    def load_data(self):

        return pd.read_csv(self.link_to_corpus)

    def clean_corpus(self):

        data = self.load_data()
        cleaned_corpus = []
        clean_df = pd.DataFrame()
        flag = ('label' in data.columns)

        # Looping over articles to clean them
        for article in range(len(data['text'])):
            if type(data['text'][article]) == str:
                removed_numbers_text = re.sub(r'[0-9]+', '', data['text'][article])
                filter_punctuations = " ".join(re.findall(r'\w+', removed_numbers_text))
                remove_noise = re.sub(r"\b[a-zA-Z]\b", "", filter_punctuations)
                clean_string = ' '.join([word for word in remove_noise.split() if word not in self.stopwords_dict])
                if flag:
                    cleaned_corpus.append((clean_string.lower(),  data['id'][article], data['label'][article]))
                else:
                    cleaned_corpus.append((clean_string.lower(), data['id'][article]))

        clean_df['text'] = [text[0] for text in cleaned_corpus]
        clean_df['id'] = [article[1] for article in cleaned_corpus]

        if flag:
            clean_df['label'] = [label[2] for label in cleaned_corpus]

        clean_df.to_csv('clean.csv')
        self.clean_dataset = clean_df

    def get_clean_data(self):
        self.clean_corpus()
        return self.clean_dataset

    def data_insights(self):

        train_dataset = self.load_data()
        print("\n--------------------------Data Insights--------------------\n")
        print(train_dataset.info())
        print("\nInstances for each class:")
        print(train_dataset['label'].value_counts())

        # Word Clouds for reliable and Unreliable Data
        reliable_train_index = [i for i, o in enumerate(train_dataset['label']) if o == 0]
        reliable_articles = np.array(train_dataset['text'])[reliable_train_index]
        wordcloud_reliable = WordCloud(background_color='black', stopwords=self.stopwords_dict.keys(), random_state=42,
                                       width=800, height=400, mask=None)
        wordcloud_reliable.generate(str(reliable_articles))
        wc_image_reliable = wordcloud_reliable.to_image()
        wc_image_reliable.show()

        unreliable_train_index = [i for i, o in enumerate(train_dataset['label']) if o == 1]
        unreliable_articles = np.array(train_dataset['text'])[unreliable_train_index]
        wordcloud_unreliable = WordCloud(background_color='black', stopwords=self.stopwords_dict.keys(), random_state=42, width=800, height=400, mask=None)
        wordcloud_unreliable.generate(str(unreliable_articles))
        wc_image_unreliable = wordcloud_unreliable.to_image()
        wc_image_unreliable.show()
