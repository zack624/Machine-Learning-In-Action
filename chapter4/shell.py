# -*- coding: utf-8 -*-

from numpy import *
import bayes
postList,classVec = bayes.loadDataSet()
vocabList = bayes.createVocabList(postList)
mat = []
for i in postList:
    mat.append(bayes.setOfWords2Vec(vocabList,i))
p0,p1,pAbusive = bayes.trainNB0(mat,classVec)

import bayes
import feedparser
ny = feedparser.parse('http://feeds.bbci.co.uk/news/science_and_environment/rss.xml')
sf = feedparser.parse('http://feeds.bbci.co.uk/news/education/rss.xml')
# len(ny['entries'])
vocabList,p0,p1 = bayes.localWords(ny,sf)
