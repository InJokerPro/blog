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

