from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from preprocessing import PreprocessData
import pandas as pd
from collections import Counter
import pandas as pd
import re
from wordcloud import WordCloud
import numpy as np


class ModelPipeline:

    def __init__(self, vectorizer, classifier, dataset):
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.dataset = dataset
        self.pipeline = None

    def make_pipeline(self):
        self.pipeline = Pipeline([('tfidf', self.vectorizer), (
            'classifier', self.classifier)])
        self.pipeline.fit(self.dataset['text'], self.dataset['label'])

    def make_predictions(self, data, label):
        predicted = self.pipeline.predict(data)
        print(np.mean(predicted == label))
        print(classification_report(label, predicted, target_names=['0', '1']))


def main():

    print("Creating models.............")
    vectorizer = TfidfVectorizer()
    classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)

    print("Processing Training and Testing Data...............")
    link_to_data = '/Users/jhanviarora/Desktop/News-Reliability-Classification/fake_news/train.csv'
    stopwords_dict = Counter(stopwords.words('english'))
    traindata_process = PreprocessData(link_to_data, stopwords_dict)
    testdata_process = PreprocessData('/Users/jhanviarora/Desktop/News-Reliability-Classification/fake_news/test.csv', stopwords_dict)
    training_data = traindata_process.get_clean_data()
    test_data = testdata_process.get_clean_data()

    print("\nGenerating training data insights.........")
    traindata_process.data_insights()

    print("\nTraining model.......")
    model_pipeline = ModelPipeline(vectorizer, classifier, training_data)
    model_pipeline.make_pipeline()

    print("\nMaking Predictions........")
    test_labels = pd.read_csv('/Users/jhanviarora/Desktop/News-Reliability-Classification/fake_news/labels.csv')
    ids_test = [x for x in test_data['id']]
    test_labels = test_labels[test_labels['id'].isin(ids_test)]
    model_pipeline.make_predictions(test_data['text'], test_labels['label'])


if __name__ == "__main__":
    main()