# Databricks notebook source
import nltk
nltk.download('stopwords')

# COMMAND ----------

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

import plotly.express as px


# COMMAND ----------

data = pd.read_csv("/dbfs/FileStore/amazon_reviews_us_Mobile_Electronics_v1_00.csv", low_memory=False)
data.head()

# COMMAND ----------

data = data[['review_body','star_rating']]
data.dropna(inplace=True)
data['star_rating'] = data['star_rating'].astype(int)

# COMMAND ----------

### Over sampling to balance Dataset

# COMMAND ----------

data['star_rating'].value_counts()

# COMMAND ----------

data['len'] = data['review_body'].apply(lambda s : len(s))

# COMMAND ----------

data  = data[data.len >1]

# COMMAND ----------

data['len'].plot.hist(bins=100)

# COMMAND ----------

data.len.quantile(0.9)

# COMMAND ----------

data.star_rating.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleaning

# COMMAND ----------

import copy
df = copy.deepcopy(data)

# COMMAND ----------

import re

def clean_text(x):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', x)
    return x

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x
def remove_punctuation(x):
  new_string = re.sub(r'[^\w\s]', '', x)
  return new_string

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

# COMMAND ----------

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re
contractions, contractions_re = _get_contractions(contraction_dict)
def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)
# Usage
replace_contractions("this's a text with contraction")

# COMMAND ----------

# lower the text
df["review_body"] = df["review_body"].apply(lambda x: x.lower())

# Clean the text
df["review_body"] = df["review_body"].apply(lambda x: clean_text(x))

# Clean numbers
df["review_body"] = df["review_body"].apply(lambda x: clean_numbers(x))

# Clean Contractions
df["review_body"] = df["review_body"].apply(lambda x: replace_contractions(x))

df["review_body"] = df["review_body"].apply(lambda x: remove_punctuation(x))

# COMMAND ----------

count_df = df[['review_body','star_rating']].groupby('review_body').aggregate({'star_rating':'count'}).reset_index().sort_values('star_rating',ascending=False)
count_df.head(120)

# COMMAND ----------

px.bar(count_df[count_df['star_rating']>4],x='review_body',y='star_rating')

# COMMAND ----------

df['len'].max(),df['len'].min(), df['len'].mean()

# COMMAND ----------

df['len'].describe()

# COMMAND ----------

df['len'].plot.kde()

# COMMAND ----------

df.drop("len",inplace = True, axis=1)

# COMMAND ----------

df.star_rating.value_counts()

# COMMAND ----------

df1= pd.get_dummies(df['star_rating'])
df2 = pd.concat([df,df1],axis=1)
df2.drop("star_rating", inplace = True, axis=1)
df2.head()

# COMMAND ----------

df2.columns

# COMMAND ----------

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size

df2[[ 1, 2, 3, 4, 5]].sum(axis=0).plot.bar()

# COMMAND ----------

X = []
sentences = list(df["review_body"])
for sen in sentences:
    X.append(preprocess_text(sen))

y = df2[[ 1, 2, 3, 4, 5]].values

# COMMAND ----------

type(y[0])

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# COMMAND ----------

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# COMMAND ----------

y_train

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://nlp.stanford.edu/data/glove.6B.zip -P /dbfs/tmp/glove

# COMMAND ----------

dbutils.fs.ls("/tmp/glove")

# COMMAND ----------

import zipfile
# EXTRACT if Necessary
# Unzip and extract in file. Maybe, could be better to unzip in memory with StringIO.
with zipfile.ZipFile('/dbfs/tmp/glove/glove.6B.zip', 'r') as zip_ref:
    zip_ref.extractall('/dbfs/tmp/glove/')

# COMMAND ----------

dbutils.fs.ls("/tmp/glove")

# COMMAND ----------

from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open('/dbfs/tmp/glove/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# COMMAND ----------

deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(5, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# COMMAND ----------

print(model.summary())

# COMMAND ----------

history = model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2)


# COMMAND ----------

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# COMMAND ----------

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
