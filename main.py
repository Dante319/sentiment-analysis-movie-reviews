#import statements
import csv
import glob
import pandas as pd
import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from summa.summarizer import summarize


#Movie Reviews Summarization
def text_summarize():
    neg = "C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/test/neg/"
    pos = "C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/test/pos/"
    outfile = open("C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/train/res.csv", "w+", newline='')
    count=0

    for files in glob.glob(neg +"*.txt"):
        count = count+1
        infile = open(files, errors='ignore')
        text = infile.read()
        res = summarize(text, ratio=0.2)
        temp = ""
        for line in res:
            temp += line.rstrip('\n')
        CSVWriter = csv.writer(outfile)
        CSVWriter.writerow(['0', str(temp)]) #0 = Negative Review

    for files in glob.glob(pos +"*.txt"):
        count = count+1
        infile = open(files, errors='ignore')
        text = infile.read()
        res = summarize(text, ratio=0.2)
        temp = ""
        for line in res:
            temp += line.rstrip('\n')
        CSVWriter = csv.writer(outfile)
        CSVWriter.writerow(['1', str(temp)]) #1 = Positive Review #This writes a row with label and summarized review to a csv file


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def rem_punctuation(text):
    punc_removed = text.translate(str.maketrans('', '', string.punctuation))
    return punc_removed

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas


def preprocess():
    datafile = "C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/test/test_res.csv"

    texts = pd.read_csv(datafile, encoding='latin1', header=None)
    #texts = texts[:200]
    #labels = texts.iloc[:,0]
    #raw_text = texts.iloc[:,1]
    #print(raw_text)

    outfile = "C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/test/test_process_res.csv"
    f_out = open(outfile, mode='w+', newline='')
    write = csv.writer(f_out, quotechar='"')

    for index, row in texts.iterrows():
        text = row[1]
        label = row[0]
        html_free = strip_html(text)
        words = nltk.word_tokenize(html_free)
        words = normalize(words)
        stems, lemmas = stem_and_lemmatize(words)
        normalized = " ".join(lemmas)
        if(normalized != ''):
            write.writerow(['{}'.format(label), '{}'.format(normalized)])


#Sentiment Analysis using Logistic Regression
def log_reg():
    reviews_train_clean = []
    datafile = "C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/train/train_process_res.csv"
    texts = pd.read_csv(datafile, encoding='latin1', header=None)
    for index, row in texts.iterrows():
        line = row[1]
        #print(line)
        reviews_train_clean.append(line.strip())

    reviews_test_clean = []

    datafile = "C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/test/test_process_res.csv"
    texts = pd.read_csv(datafile, encoding='latin1', header=None)
    for index, row in texts.iterrows():
        line = row[1]
        # print(line)
        reviews_test_clean.append(line.strip())

    cv = CountVectorizer(binary=True)
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)

    #print(X)

    train_target = [0 if i < 10804 else 1 for i in range(21126)]
    test_target = [0 if i < 10679 else 1 for i in range(21126)]

    X_train, X_val, y_train, y_val = train_test_split(
        X, train_target, train_size=0.75
    )

    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        lr = LogisticRegression(C=c)
        lr.fit(X_train, y_train)
        print("Accuracy for C=%s: %s"
              % (c, accuracy_score(y_val, lr.predict(X_val))))

    final_model = LogisticRegression(C=0.25)
    final_model.fit(X, train_target)
    res = final_model.predict(X_test)
    print(len(res))
    print(test_target)
    print ("Final Accuracy: %s"% accuracy_score(test_target, final_model.predict(X_test)))

#Sentiment Analysis using Naive Bayes Classifier
def NB():
    def create_word_features(words):
        my_dict = dict([(word, True) for word in words])
        return my_dict

    def words(text):
        return

    train_neg_reviews = []
    train_pos_reviews = []
    datafile = "C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/train/train_process_res.csv"
    texts = pd.read_csv(datafile, encoding='latin1', header=None)
    for index, row in texts.iterrows():
        line = row[1]
        label = row[0]
        #print(line)
        if(label==0):
            words = word_tokenize(line)
            #print(words)
            train_neg_reviews.append((create_word_features(words), "negative"))
        else:
            words = word_tokenize(line)
            train_pos_reviews.append((create_word_features(words), "positive"))

    print(len(train_neg_reviews))
    print(len(train_pos_reviews))

    test_neg_reviews = []
    test_pos_reviews = []
    datafile = "C:/Users/Dante/PycharmProjects/Compiler/Datasets/aclImdb/test/test_process_res.csv"
    texts = pd.read_csv(datafile, encoding='latin1', header=None)
    for index, row in texts.iterrows():
        line = row[1]
        label = row[0]
        # print(line)
        if (label == 0):
            words = word_tokenize(line)
            # print(words)
            test_neg_reviews.append((create_word_features(words), "negative"))
        else:
            words = word_tokenize(line)
            test_pos_reviews.append((create_word_features(words), "positive"))

    print(len(test_neg_reviews))
    print(len(test_pos_reviews))

    train_set = train_neg_reviews + train_pos_reviews
    test_set = test_neg_reviews + test_pos_reviews
    print(len(train_set), len(test_set))

    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = nltk.classify.util.accuracy(classifier, test_set)
    print(accuracy * 100)

log_reg()
NB()
