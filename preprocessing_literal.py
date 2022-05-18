import rdflib
from rdflib import Graph
import codecs
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re


from builtins import bytes, range

import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="himalaya.ttf",size=20)

import numpy as np

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def dataType(string):
    odp='string'
    patternBIT=re.compile('[01]') # bit
    patternINT=re.compile('[0-9]+') # int
    patternFLOAT=re.compile('[0-9]+\.[0-9]+') # float
    patternTEXT=re.compile('[a-zA-Z0-9]+') # text
    if patternTEXT.match(string):
        odp= "string"
    if patternINT.match(string):
        odp= "integer"
    if patternFLOAT.match(string):
        odp= "float"
    return odp

def getRDFData(o):
    # 判断是否是URIRef类型，还是rdflib.term.Literal类型
    if isinstance(o, rdflib.term.URIRef):
        data_type = "uri"
    else:
        data_type = o.datatype  # 得到三元组类型
        if data_type == None:
            data_type = dataType(o) # 通过dataType方法得到具体数据类型
        # rdflib.term.Literal存在具体dataType，
        # 例如<http://yago-knowledge.org/resource/Eric_Carruthers_(footballer)>
        # <http://dbpedia.org/ontology/birthDate> "1953-02-02"^^<http://www.w3.org/2001/XMLSchema#date> .
        else:
            if "#" in o.datatype:
                data_type = o.datatype.split('#')[1].lower() # 以#进行分割, 得到第二部分，例如得到date
            else:
                data_type = dataType(o)
        ## 对时间特殊处理
        if data_type == 'gmonthday' or data_type=='gyear':
            data_type = 'date'
        # 对positiveinteger, nonnegativeinteger特殊处理
        if data_type == 'positiveinteger' or data_type == 'int' or data_type == 'nonnegativeinteger':
            data_type = 'integer'
    return o, data_type
def getAllLiteral(source1, source2, target):
    # 提出文档的所有字符
    yago_filename = 'dataset/DWY-NB/DY-NB/' + source1
    dbp_filename = 'dataset/DWY-NB/DY-NB/' + source2

    target = codecs.open(
        'data/' + target, 'w',
        encoding="utf8")

    # 创建一个图谱
    graph = Graph()
    graph.parse(location=yago_filename, format='nt')
    graph.parse(location=dbp_filename, format='nt')

    for s, p, o in graph:
        o = getRDFData(o)
        if isinstance(o[0], rdflib.term.Literal):
            target.writelines((str)(o[0]).encode('utf-8').decode('unicode_escape'))
            target.write('\n')  # 显示写入换行
    target.close()

# Word2Vec第一个参数代表要训练的语料
# sg=1 表示使用Skip-Gram模型进行训练
# size 表示特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
# window 表示当前词与预测词在一个句子中的最大距离是多少
# min_count 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
# workers 表示训练的并行数
# sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
def word2vec(fileName):
    # 首先打开需要训练的文本
    # sentences = word2vec.LineSentence('D:/StudyFile/ProjectWorkstation/EA/EntityAlignment/dataset/segment/in_the_name_of_people_segment.txt')
    sentences = open(fileName, 'rb')
    # 通过Word2vec进行训练
    model = Word2Vec(LineSentence(sentences), sg=1, vector_size=10, window=10, min_count=1, workers=15, sample=1e-3)
    # 保存训练好的模型
    model.save(fileName + '.word2vec')

    print('训练完成')


# 词向量可视化
def tsne_plot(model, words_num):
    labels = []
    tokens = []
    for word in model.wv.index_to_key:
        tokens.append(model.wv[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    plt.figure(figsize=(10, 10))
    for i in range(words_num):
        plt.scatter(x[i], y[i])
        if b'\xe0' in bytes(labels[i], encoding="utf-8"):
            this_font = font
        else:
            this_font = 'SimHei'
        plt.annotate(labels[i],
                     Fontproperties=this_font,
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

def getLiteralArrayByWord2vec(o, model):
    literal_object = np.zeros(20)
    strs =  o.split(' ')
    for str in strs:
        literal_object =  literal_object + model.wv[str]
    return literal_object

if __name__ == '__main__':
    # getAllLiteral('yago.ttl', 'dbp_yago.ttl', 'yago_dbp_literal_all.txt')
    fileName = 'D:/Project/EA/EA-Multi-Dimensional-Archives/data/yago_dbp_literal_all.txt'
    word2vec(fileName)
    # 加载模型
    # model = Word2Vec.load('D:/Project/EA/EA-Multi-Dimensional-Archives/data/yago_dbp_literal_all.txt.word2vec')
    # 获取预料数量
    # print(model.corpus_count)
    # tsne_plot(model, 50)
    # o = 'man'
    # print(getLiteralArrayByWord2vec(o, model))

