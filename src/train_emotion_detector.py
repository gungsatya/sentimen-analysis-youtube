import pandas as pd
import numpy as np
import os
import pickle

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.models import Sequential
from tf_keras.layers import Bidirectional, Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from tf_keras.callbacks import EarlyStopping


from utils.words import clean_text
from config import root_dir



embedding_path = os.path.join(root_dir, "data\\cc.id.300.vec")
file_path =  os.path.join(root_dir,"data\\Twitter_Emotion_Dataset.csv")

# 1. Load Dataset
df = pd.read_csv(file_path)

df['tweet'] = df['tweet'].apply(clean_text)

# 2. Tokenization
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300 # FastText biasanya menggunakan 300 dimensi

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df['tweet'].values)
word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(df['tweet'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# 3. Load FastText Pre-trained Vectors
# Unduh 'cc.id.300.vec' dari website FastText


embeddings_index = {}
print("Loading FastText Model...")
with open(embedding_path, encoding='utf8') as f:
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# 4. Membuat Embedding Matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 5. Build Model
model = Sequential()
model.add(Embedding(len(word_index) + 1, 
                    EMBEDDING_DIM, 
                    weights=[embedding_matrix], 
                    input_length=MAX_SEQUENCE_LENGTH, 
                    trainable=False)) # Set False agar bobot FastText tidak berubah
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(df['label'].unique()), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6. Training
# Pastikan label sudah di-encode (sama seperti skrip sebelumnya)
Y = pd.get_dummies(df['label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model.fit(X_train, Y_train, 
          epochs=15, 
          batch_size=64, 
          validation_split=0.1,
          callbacks=[EarlyStopping(monitor='val_loss', patience=3)])


model.save(os.path.join(root_dir, "models\\model_emosi_fasttext.h5"))

with open(os.path.join(root_dir, "models\\tokenizer.pickle"), 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model dan Tokenizer berhasil disimpan!")