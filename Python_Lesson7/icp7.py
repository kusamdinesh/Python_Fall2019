# Task-1: Doing Http request to a URL and doing HTML parsing using BeautifulSoup --> Storing in the txt document file
from bs4 import BeautifulSoup
import requests

page_link = 'https://en.wikipedia.org/wiki/Google'
page_response = requests.get(page_link)

page_content = BeautifulSoup(page_response.content, "html.parser")
text = page_content.get_text()

# Task-2: Storing the text output to a file
with open('input.txt', 'w', encoding='utf-8') as f:
    f.write(text)

# Task-3(A): Tokenizing the text output

f = open('input.txt', 'r', encoding='utf-8')
input = f.read()

input = "This is a sentence. Not a stop word and stemmer"

import nltk

nltk.download('punkt')

stokens = nltk.sent_tokenize(input)
wtokens = nltk.word_tokenize(input)

for s in stokens:
    print(s)

print('\n')

for t in wtokens:
    print(t)

# Task-3(B): Parts of Speech Tagging on the text output
nltk.download('averaged_perceptron_tagger')

print(nltk.pos_tag(wtokens))

# Task-3(C): Stemming
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer

ps = PorterStemmer()
sb = SnowballStemmer('english')
lc = LancasterStemmer()
for x in wtokens:
    print(ps.stem(x))
    print(sb.stem(x))
    print(lc.stem(x))

# Task-3(D): Lemmatization
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

wtokens = nltk.word_tokenize(input)

lemmatizer = WordNetLemmatizer()

for x in wtokens:
    print(lemmatizer.lemmatize(x))

# Task-3(E): Trigram
nltk.download('maxent_ne_chunker')
nltk.download('words')

stokens = nltk.sent_tokenize(input)

for sent in stokens:
    wtokens = nltk.word_tokenize(sent)
    for j in range(len(wtokens) - 2):
        print(wtokens[j], wtokens[j + 1], wtokens[j + 2])

# Task-3(F): Named Entity Recognition
from nltk import wordpunct_tokenize, pos_tag, ne_chunk

stokens = nltk.sent_tokenize(input)

for i in stokens:
    print(ne_chunk(pos_tag(wordpunct_tokenize(i))))

# Task-4(A): Using KNeighborsClassifier and compare the accuracy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)

clf = KNeighborsClassifier(n_neighbors=3)
# clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("KNeighborsClassifier Score",score)

# Task-4(B): Changing TF_IDF Vectorizer to use bi-grams and compare the accuracy
from sklearn.naive_bayes import MultinomialNB

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("Score  after changing TF_IDF Vectorizer to use bi-grams",score)

# Task-4(C): Removing stop words and then comparing the accuracy
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print("Score After Removing stop words",score)