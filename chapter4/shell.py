# -*- coding: utf-8 -*-

# 训练数据
from numpy import *
import bayes
postList,classVec = bayes.loadDataSet()
vocabList = bayes.createVocabList(postList)
mat = []
for i in postList:
    mat.append(bayes.setOfWords2Vec(vocabList,i))
p0,p1,pAbusive = bayes.trainNB0(mat,classVec)

# 利用rss源文档测试
import bayes
import feedparser
sci_env = feedparser.parse('http://feeds.bbci.co.uk/news/science_and_environment/rss.xml')
edu = feedparser.parse('http://feeds.bbci.co.uk/news/education/rss.xml')
rate = 0.0
for i in range(10):
    vocabList,p0,p1,erate = bayes.localWords(sci_env,edu)
    rate += erate

print "error rate: %f" % (rate/10)
# len(ny['entries'])

# 获取各分类文档的出现次数最多的词
import bayes
import feedparser
sci_env = feedparser.parse('http://feeds.bbci.co.uk/news/science_and_environment/rss.xml')
edu = feedparser.parse('http://feeds.bbci.co.uk/news/education/rss.xml')
bayes.getTopWords(sci_env,edu)