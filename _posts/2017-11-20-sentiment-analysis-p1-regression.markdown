---
layout: post
title:  "ML4Noobs Part 1.0: Sentiment regressor"
img: china_tiger_leaping_gorge.jpg
date:   2017-11-20 00:00:00 +0200
description: A noob's guide to sentiment analysis with machine learning
---

## Introduction : 

Sentiment analysis has always been an important usage of machine learning. And there are a lot of ways to frame this problem so that they can be calculated in a certain way. In this series, we will be talking about 2 approaches to solve the sentiment analysis problem. The dataset used in this example is called [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews).
As always, before deciding what to do with the dataset, we should first take a look at it.
```
import numpy as np 
import pandas as pd 

df_train = pd.read_table('train.tsv')
df_train.head()
```
You should see a dataframe like this : 

![](sentiment_analysis/df_train.jpg)

The sentiment labels are:
0 - negative 
1 - somewhat negative 
2 - neutral 
3 - somewhat positive 
4 - positive

There are 156060 samples in total.If we look at the label now. It's obvious that it's ranked by 'happiness' or 'positivity'. So with this kind of label, it's very easy to think of it as a kind of regression problem, where our target is the degree of 'happiness' or 'positivity' that the sentences express. So we will treat it as a regression problem for now.  
  
Too make a regression model with this data, the easiest way is to simply count how many words appears in the sentences. And correctly give each word their own weights so that they can contribute to the sentences' "happiness". They can make both positive and negative contributions. At the end we simply add up all the contributions each word in the sentence has made, we have the "happiness" for that sentence.

#### Preprocessing :

Preprocessing :
In order to correctly represent our data in a mathematical way, we need to first "convert" the spoken word into some kind of numeric way so that we can do math with it. In this example, for the sake of memory, we can use an interger as identifier for a certain word. For instance, when we have phase _I wanna play games_ ,we can apply an interger for each word like this :
```
{I : 0,wanna : 1, play : 2,games : 3}
```
Hmm... Looks good. Probably there's no wrong with it. But consider this : Does the word _wanna_ have to follow _I_? Not exactly. In fact, this is a very bad idea. Because we have manually ranked the words so they some certain relationship to each other. Not that I said they're not related at all. But not like this, at least. Further talking about relationships between words is another subject which we don't wanna talk about for now. We only want a independent representations of words so that we can work on our sentences.
So how about this :
```
{I : (1,0,0,0) . wanna : (0,1,0,0) , play : (0,0,1,0) , games : (0,0,0,1)}
```
This is actually a lot better than the last one. And this is how we can numerically identify a word without ranking them. This is the called one-hot encoding, where we initialize a vector with a 1 in it's unique slot while 0 in all the other slots.
Now that we have successfully encoded the words. But there's still a small problem. That is, some of the words don't have any contributions to "happiness". For instance, the word "I" is just a mean to tell that it's a person. I doesn't contribute to the sentences mood. And there are a lot of this kind of words. They're called "stopwords". We need to get rid of these words to saves memory and maybe prevent overfitting.
So the final representation of words looks like this :
```
{wanna : (1,0,0) , play : (0,1,0) , games : (0,0,1)}
```
The code below shows the process of giving each word an interger as an index to tell which number to be set hot :
```
phrases = ''
for i in range(len(df_train['Phrase'])) :
    phrases += df_train['Phrase'].loc[i] + ' '
words = phrases.split()
stopwords = 'a an and are as at be by for trom has he in is it its of on that the to was were will with they them she her him'
stopwords = stopwords.split()
words = [w.lower() for w in words if w not in [',','.','!','?','\'s']]
words = [w for w in words if w not in stopwords]
```
I'm only trying to demostrate what the process looks like. So it might not cover all the stop words. But you can google for some more and add them to the list. It's a very easy task. Another reason is that I wanna show you we can try to get rid of stop words by only filtering out the words that are not in our desired number of presence. We can also get rid of words that rarely used, like terminologies.
```
from collections import Counter
counter = Counter(words).most_common()
print('Top 10 frequency : ',counter[:10])
print('Bottom 10 frequency : ',counter[-10:])
```
If you're using the same dataset, you should see that it's not doing a great job in getting rid of stopwords. And it sort of filtered out a lot of useful informations at the same time. You can try out different caps as well.

The following code shows the process of giving each word their own index, and writing a generator so that we can train them on batch.

_There's a bug in the generator, haven't fix it yet, but you can get away with a try - except in the fitting process_

```
words = set([x[0] for x in counter])
words_enum = enumerate(words)
word_dict = { w : i for i,w in words_enum }
vector_len=len(words)

def batch_sentences(orig_df,batch_size) : 
    while True : 
        idx=0
        sentence_batch = []
        sentiment_batch = []
        
        for i in range(len(orig_df)) : 
            key_words = np.array([0]*vector_len)
            sentences = orig_df['Phrase'].iloc[i]
            words_in_sentence = [x for x in sentences.split() if x in words]
            for w in words_in_sentence : 
                key_words[word_dict[w]] += 1
            sentence_batch.append(key_words)
            sentiment_batch.append(orig_df['Sentiment'].iloc[i])
            idx += 1
            if idx % batch_size == 0 : 
                yield np.array(sentence_batch),np.array(sentiment_batch)
                sentence_batch = []
                sentiment_batch = []
        
        yield np.array(sentence_batch),np.array(sentiment_batch)
```

A single layer neural net can do exactly what we want here : applying different weights, optimizing the weights and adding those contributions up. We're not about to discuss optimization strategies here. So we will simply use stochastic gradient descent provided by keras with it's default parameters here.

```
from keras.layers import Dense,Input
from keras.models import Model
Inputs = Input(shape=(vector_len,))
predictions = Dense(1,activation='linear')(Inputs)
model=Model(inputs=Inputs,outputs=predictions)
model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
train_limit = 130000
batch_size = 10000
model.fit_generator(generator=batch_sentences(df_train.loc[:train_limit],batch_size),
                   steps_per_epoch=np.ceil(train_limit/batch_size),
                   validation_data=batch_sentences(df_train.loc[train_limit:],batch_size),
                   validation_steps=np.ceil((len(df_train)-train_limit)/batch_size),
                   epochs=5)
```

After fitting our model to the data. We can now test how well our regression works.

```
sent = 'Fantastic movie . It deserves more credits'

sent = sent.split()
sent = [word_dict[x.lower()] for x in sent if x in words]

kw = np.array([0]*vector_len)
for i in sent : 
    kw[i] += 1
kw = kw.reshape((-1,vector_len))
happiness = model.predict(kw)
print(happiness)
```
In my case, this model is still pretty stupid, giving only 1.92 'happiness'... But anyway, you get the idea. Try to change a dataset or tweak the values and see if you can get a better result. Personally, I think this dataset is still not enough for an nlp problem.