from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values
print(np.unique(y))  # --> Printing unique labels i.e; 3 labels as {neg, pos, unsup}

#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data

sentences = tokenizer.texts_to_matrix(sentences)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Number of features
# print(input_dim)
model = Sequential()
model.add(layers.Dense(300, activation='relu', input_dim=2000)) ## Inputdeimension is missing
model.add(layers.Dense(3, activation='softmax')) # mistake2: Output layer should have 3 neurons as the outputs are 3 {neg,pos, unsup}
                                                 # mistake3: The output activation function sould be softmax. becusae this will forces the output
                                                 # of neural network to represent the probability that the input falss into each of the classess.
                                                 # without the softmax the node's ouput are simply numeric values which is not true in this case.
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=2, verbose=True, validation_data=(X_test,y_test), batch_size=256)