---
layout: post
comments: true
title:  "Building a URL classifier using DMOZ data"
excerpt: "We will look at CNNs and how to use them with word embeddings."
date:   2017-01-18 15:00:00
---

<img src="/assets/url/grab1.png">

[DMOZ](https://www.dmoz.org/) (from directory.mozilla.org, an earlier domain name) is a multilingual open-content directory of World Wide Web links. The site and community who maintain it are also known as the Open Directory Project (ODP). It is owned by AOL but constructed and maintained by a community of volunteer editors.

First, we would download a dump of their open directory from here [contents](http://rdf.dmoz.org/).
The content file has a nice structure and we would take advantage of it to extract useful data.

Topics are under 13 main categories which further have lots of subcategories.

Here are the root categories -
<img src="/assets/url/grab2.png" width="100%" />

Subcategories under health -
<img src="/assets/url/grab3.png" width="100%" />

For our URL classification problem, we will only focus on the root heirarchy.

Below is the highlight of what we are looking for in the contents file.
<img src="/assets/url/grab.png" width="100%" />

### Import modules

```
import os
import sys
import re
import pandas as pd
import time
import numpy as np
import pickle
from collections import Counter

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.layer_utils import layer_from_config
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold, train_test_split
import operator
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
%matplotlib inline
```
>Using Theano backend.
Using gpu device 0: GeForce GTX 980 (CNMeM is enabled with initial size: 70.0% of memory, cuDNN 5005)

### Read data
We will create an empty dataframe here which we would use afterwards to store `categories`, `titles` and `descriptions` of URLs.

```
path = '/home/dcrush/Documents/Url Classifier'
fn = 'content.rdf.u8'

# initialise empty dataframe
df = pd.DataFrame(columns=['category', 'title', 'desc'])

# we will read everything in one go and then filter out useful stuff
with open(os.path.join(path, fn), 'r') as fl_in:
    lines = fl_in.readlines()
```

### Filter titles, descriptions and topics

```
lines = [str(line) for line in lines]
titles = [re.findall('<d:Title>(.+)</d:Title>', line) for line in lines]
descs = [re.findall('<d:Description>(.+)</d:Description>', line) for line in lines]
topics = [re.findall('<topic>(.+)</topic>', line) for line in lines]
del lines
```

Let's check positions as well as counts of `titles`, `descriptions` and `topics`.

```
titles_pos = [i for i, x in enumerate(titles) if len(x)>0]
descs_pos = [i for i, x in enumerate(descs) if len(x)>0]
topics_pos = [i for i, x in enumerate(topics) if len(x)>0]

print '# titles found', len(titles_pos)
print '# descriptions found', len(descs_pos)
print '# topics found', len(topics_pos)
```
>titles found 3579877
descriptions found 3578583
topics found 3579877

Ideally, lengths of `titles_pos`, `descs_pos` and `topics_pos` should all be same. Here we have some missing descriptions.
Let's filter out only those cases where occupied index of descriptions is `titles_index`+1 and occupied index of topics is `titles_index`+2.

```
topics_list = []
titles_list = []
descs_list = []
for line_counter, i in enumerate(titles_pos):
    if line_counter % 1000000 == 0:
        print line_counter, 'processed'
    if len(descs[i+1])>0 and len(topics[i+2])>0:
        topics_list.append(topics[i+2][0].split('/')[1])
        titles_list.append(titles[i][0])
        descs_list.append(descs[i+1][0])
```
>0 processed
1000000 processed
2000000 processed
3000000 processed

### Write to dataframe

```
df.category = topics_list
df.title = titles_list
df.desc = descs_list
df.shape
```
>(3534499, 3)

```
df.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>title</th>
      <th>desc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Arts</td>
      <td>About.com: Animation Guide</td>
      <td>Keep up with developments in online animation ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Arts</td>
      <td>Toonhound</td>
      <td>British cartoon, animation and comic strip cre...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arts</td>
      <td>Digital Media FX: The History of Animation</td>
      <td>Michael Crandol takes an exhaustive look at th...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arts</td>
      <td>Richard's Animated Divots</td>
      <td>Chronology of animated movies, television prog...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Arts</td>
      <td>Nini's Bishonen Dungeon</td>
      <td>Shrines to Vega, Taiki, Dilandau, and Tiger Ey...</td>
    </tr>
  </tbody>
</table>
</div>

```
df.tail()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>title</th>
      <th>desc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3534494</th>
      <td>World</td>
      <td>Tébabet tor békiti</td>
      <td>Saghlamliq we tébabet heqqide. Eng yéngi yolla...</td>
    </tr>
    <tr>
      <th>3534495</th>
      <td>World</td>
      <td>Tengritagh uyghur tori</td>
      <td>Shinjang uyghur aptonom rayonluq xelq hökümiti...</td>
    </tr>
    <tr>
      <th>3534496</th>
      <td>World</td>
      <td>Erkin asiya radiosi</td>
      <td>Xewerler, mulahizeler, köpxil wasteler. Radio ...</td>
    </tr>
    <tr>
      <th>3534497</th>
      <td>World</td>
      <td>Uyghurlar tori</td>
      <td>Aile we turmush, ata-anilar, aliy mektep we ya...</td>
    </tr>
    <tr>
      <th>3534498</th>
      <td>World</td>
      <td>Istiqlal radio-tilivizisi</td>
      <td>Sherqi türküstan xewerliri, türk dünyasi xewer...</td>
    </tr>
  </tbody>
</table>
</div>

Some of the stuff here is not english.

Let's explore the distribution of categories we are dealing with.

```
plt.figure(figsize=(12, 5))
df.category.value_counts().plot(kind='bar');
plt.title('Category counts');
```
<img src="/assets/url/output_20_0.png" width="100%" />

We will drop `World` and `Regional` from our model as they have multiple langauges as well as they make the data very skewed thus making our job difficult :)

```
print 'df shape before', df.shape
df = df[~df.category.isin(['World', 'Regional'])]
print 'df shape after', df.shape
```
<img src="/assets/url/output_23_0.png" width="100%" />

```
plt.figure(figsize=(12, 5))
df.category.value_counts().plot(kind='bar');
plt.title('Category counts');
```

There is still a lot of class imbalance and this will surely affect our model. We are going to ignore this but there are many ways to handle this -

* Get more data for classes with less data
* Undersample `Business` and `Society`
* Create synthetic data

We are going to model a `CNN` using word embeddings which we are going to build using the `descriptions`. To that end, we will concat `title` and `description`.

```
df.desc = df.title + ' ' + df.desc
df = df.drop(['title'], axis=1)
```
```
df.desc = df.desc.str.lower()
```

Let's explore the lengths of descriptions we are looking at

```
lens = [len(x) for x in df.desc]
plt.figure(figsize=(12, 5));
print max(lens), min(lens), np.mean(lens)
sns.distplot(lens);
plt.title('Description length distribution');
```
>978 10 131.182213298

<img src="/assets/url/output_29_1.png" width="100%" />

### Word dictionary
We need to create a word dictionary which would be used for word-id mapping.
We will limit description length to 200 words. So when testing, we can only use max 200 words.
Also, we will limit our vocabulary to 5000 words.

```
vocab_size = 5000
seq_len = 200
```
```
words = [re.findall('[\w\d]+', x) for x in df.desc]
all_words = []
for x in words:
    all_words += x
word_to_id = Counter(all_words).most_common(vocab_size)
```

Top 10 most frequent words

```
word_to_id[:10]
```
>    [('and', 1271555),
     ('of', 575053),
     ('the', 507656),
     ('in', 298967),
     ('a', 246295),
     ('for', 231537),
     ('information', 183775),
     ('to', 183742),
     ('on', 127852),
     ('s', 110210)]

Least common words in our vocab

```
word_to_id[-10:]
```
>    [('directed', 384),
     ('outsourcing', 384),
     ('arbor', 384),
     ('roof', 383),
     ('perry', 383),
     ('hungary', 383),
     ('psi', 383),
     ('theta', 383),
     ('operate', 383),
     ('whether', 382)]

These words don't seem to be that rare. We could increase our vocabulary size but we won't as it would increase our training time as well.

We don't care about the counts, so we will create a dictionary for these 5000 words with values in ascending order of count

```
word_to_id = {x[0]:i for i, x in enumerate(word_to_id)}
```
```
train = [np.array([word_to_id[y] if y in word_to_id else vocab_size-1\
        for y in x]) for x in words]
```

Pad cases with length less than `seq_len`

```
train = sequence.pad_sequences(train, maxlen=seq_len, value=0)
```
```
train = train.astype('float32')
```

### CNN
```
le = LabelEncoder()
le.fit(df.category)
y = le.transform(df.category)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, \
        random_state=42)
```
```
model = Sequential([
        Embedding(vocab_size, 50, input_length=seq_len),
        Dropout(0.1),
        Convolution1D(64, 5, border_mode='same', activation='relu'),
        Dropout(0.2),
        Convolution1D(128, 5, border_mode='same', activation='relu'),
        Dropout(0.2),
        MaxPooling1D(),
        Flatten(),
        Dense(200, activation='relu'),
        Dropout(0.3),
        Dense(13, activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer=Adam(), \
        metrics=['accuracy'])
```
```
model.fit(X_train, y_train, validation_data=(X_test, y_test), \
        nb_epoch=3, batch_size=64)
```
> Train on 784856 samples, validate on 196214 samples
    Epoch 1/3
    784856/784856 [==============================] - 138s - loss: 0.9171 - acc: 0.7172 - val_loss: 0.7284 - val_acc: 0.7727
    Epoch 2/3
    784856/784856 [==============================] - 138s - loss: 0.7447 - acc: 0.7678 - val_loss: 0.6952 - val_acc: 0.7798
    Epoch 3/3
    784856/784856 [==============================] - 140s - loss: 0.7101 - acc: 0.7765 - val_loss: 0.6778 - val_acc: 0.7834

```
model.optimizer.lr = 1e-4
```

```
model.fit(X_train, y_train, validation_data=(X_test, y_test), \
        nb_epoch=1, batch_size=64)
```
>Train on 784856 samples, validate on 196214 samples
    Epoch 1/1
    784856/784856 [==============================] - 140s - loss: 0.6798 - acc: 0.7839 - val_loss: 0.6573 - val_acc: 0.7901

### Check accuracy of each category

```
y_test_1 = [np.argmax(x) for x in y_test]
y_test_1_pred = [np.argmax(x) for x in model.predict(X_test)]
```
```
temp = pd.DataFrame({'actuals':y_test_1, 'preds':y_test_1_pred})
act_pred = temp.groupby(['actuals', 'preds']).size().reset_index()
act_pred.columns = ['actual', 'preds', 'true']
```
```
act_totals = pd.DataFrame(pd.Series(y_test_1).value_counts()).reset_index()
act_totals.columns = ['actual', 'total']
```
```
final = act_pred[act_pred.actual==act_pred.preds].merge(act_totals, how='left', \
        on='actual')
```
```
final['perc_correct'] = final.true/final.total*100
final['classes'] = le.classes_
final
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual</th>
      <th>preds</th>
      <th>true</th>
      <th>total</th>
      <th>perc_correct</th>
      <th>classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>27561</td>
      <td>31707</td>
      <td>86.924023</td>
      <td>Arts</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>29007</td>
      <td>34333</td>
      <td>84.487228</td>
      <td>Business</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>12486</td>
      <td>15313</td>
      <td>81.538562</td>
      <td>Computers</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>3511</td>
      <td>5062</td>
      <td>69.359937</td>
      <td>Games</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>6418</td>
      <td>8166</td>
      <td>78.594171</td>
      <td>Health</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>5</td>
      <td>2408</td>
      <td>3630</td>
      <td>66.336088</td>
      <td>Home</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
      <td>630</td>
      <td>1212</td>
      <td>51.980198</td>
      <td>News</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
      <td>9802</td>
      <td>13700</td>
      <td>71.547445</td>
      <td>Recreation</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>8</td>
      <td>4711</td>
      <td>8096</td>
      <td>58.189229</td>
      <td>Reference</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>9</td>
      <td>10684</td>
      <td>15715</td>
      <td>67.986001</td>
      <td>Science</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>10</td>
      <td>8260</td>
      <td>12107</td>
      <td>68.224994</td>
      <td>Shopping</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>11</td>
      <td>27094</td>
      <td>32864</td>
      <td>82.442795</td>
      <td>Society</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>12</td>
      <td>12457</td>
      <td>14309</td>
      <td>87.057097</td>
      <td>Sports</td>
    </tr>
  </tbody>
</table>
</div>

### Correlation between class size and accuracy

```
from scipy.stats import pearsonr
pearsonr(final.perc, final.total)[0]
```
>0.73435146035185761

Such a high correlation between class size and accuracy clearly suggests that the class imbalance is playing a big role.

More improvements can be made by -

* Training the model for longer duration
* Resolving the class imbalance
* Using pretrained word vectors like Glove
* Make sure the model has enough variance to generalize. Usually this can be achieved by using data from other sources as well in the model

I haven't covered how to use this model for real life data. I'll do it in another post. The basic idea is to scrape content from URLs, expecially headings and paragraph tags and preprocess them using the same pipeline we used for our training.

This post has been inspired by MOOC at [Fast.ai](http://www.fast.ai/), by Jeremy Howard.
