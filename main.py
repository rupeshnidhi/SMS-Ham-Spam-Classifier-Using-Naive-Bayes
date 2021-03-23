import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


messages = pd.read_csv('./datasets/SMSSpamCollection',
                       sep='\t', names=["label", "message"])
msg_train, msg_test, label_train, label_test = train_test_split(
    messages['message'], messages['label'], test_size=0.2)

pipeline = Pipeline([
    # strings to token integer counts
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    # train on TF-IDF vectors w/ Naive Bayes classifier
    ('classifier', MultinomialNB()),
])

pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions, label_test))
