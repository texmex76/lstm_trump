from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import model_from_json
import numpy as np
import random
import sys
import io

path = get_file('trump_red.txt',
  origin='file:/home/bernhard/Documents/ml/trump_tweets_charac/trump_red.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower().replace('\n', ' ')

# create dict with chars, indices as keys, respectively
chars = sorted(list(set(text)))
print('Total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 60
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
  sentences.append(text[i: i + maxlen])
  next_chars.append(text[i + maxlen])
print('Number sentences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
  for t, char in enumerate(sentence):
    x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))
model.load_weights("model_v2_21.h5")

optimizer = RMSprop(lr=0.0001, clipvalue=0.1)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
  # helper function to sample an index from a probability array
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def on_epoch_end(epoch, _):
  epoch += 22
  model.save_weights('model_v2_' + str(epoch) + '.h5')
  print('Saved weights to disk')

  print()
  print('----- Generating text after Epoch: ' + str(epoch))

  start_index = random.randint(0, len(text) - maxlen - 1)
  for diversity in [0.2, 0.5, 1.0, 1.2]:
    print('----- diversity: ' + str(diversity))

    sentence = text[start_index: start_index + maxlen]
    sys.stdout.write(sentence)

    for i in range(180):
      x_pred = np.zeros((1, maxlen, len(chars)))
      for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.

      preds = model.predict(x_pred, verbose=0)[0]
      next_index = sample(preds, diversity)
      next_char = indices_char[next_index]

      sentence = sentence[1:] + next_char

      sys.stdout.write(next_char)
      sys.stdout.flush()
    print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
        batch_size=128,
        epochs=60,
callbacks=[print_callback])