---
layout: post
title:  "Sentiment Analysis Part I : regression"
img: china_tiger_leaping_gorge.jpg
date:   2017-11-20 00:00:00 +0200
description: A noob's guide to sentiment analysis with machine learning
---

## Introduction : 

Sentiment analysis has always been an important usage of machine learning. And there are a lot of ways to frame this problem so that they can be calculated in a certain way. In this series, we will be talking about 2 approaches to solve the sentiment analysis problem. The dataset used in this example is called Sentiment Analysis on Movie Reviews.
As always, before deciding what to do with the dataset, we should first take a look at it.

{% highlight markdown %}
import numpy as np 
import pandas as pd 

df_train = pd.read_table('train.tsv')
df_train.head()
{% endhighlight %}

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

	**_{I : 0,wanna : 1, play : 2,games : 3}_**

Hmm... Looks good. Probably there's no wrong with it. But consider this : Does the word _wanna_ have to follow _I_? Not exactly. In fact, this is a very bad idea. Because we have manually ranked the words so they some certain relationship to each other. Not that I said they're not related at all. But not like this, at least. Further talking about relationships between words is another subject which we don't wanna talk about for now. We only want a independent representations of words so that we can work on our sentences.
So how about this :

	**_{I : (1,0,0,0) . wanna : (0,1,0,0) , play : (0,0,1,0) , games : (0,0,0,1)}_**

This is actually a lot better than the last one. And this is how we can numerically identify a word without ranking them. This is the called one-hot encoding, where we initialize a vector with a 1 in it's unique slot while 0 in all the other slots.
Now that we have successfully encoded the words. But there's still a small problem. That is, some of the words don't have any contributions to "happiness". For instance, the word "I" is just a mean to tell that it's a person. I doesn't contribute to the sentences mood. And there are a lot of this kind of words. They're called "stopwords". We need to get rid of these words to saves memory and maybe prevent overfitting.
So the final representation of words looks like this :

	**_{wanna : (1,0,0) , play : (0,1,0) , games : (0,0,1)}_**

The code below shows the process of giving each word an interger as an index to tell which number to be set hot :

{% highlight markdown %}
phrases = ''

for i in range(len(df_train['Phrase'])) :
    phrases += df_train['Phrase'].loc[i] + ' '

words = phrases.split()

stopwords = 'a an and are as at be by for trom has he in is it its of on that the to was were will with they them she her him'
stopwords = stopwords.split()

words = [w.lower() for w in words if w not in [',','.','!','?','\'s']]
words = [w for w in words if w not in stopwords]
{% endhighlight %}

I'm only trying to demostrate what the process looks like.