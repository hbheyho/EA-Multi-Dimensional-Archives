#!/usr/bin/env python
# coding: utf-8


# 三元组读取和持久化.rdflib的一项主要的功能就是将一种基于语法（如xml,n3,ntriples,trix,JSON）的文件变换成一个RDF格式的知识
# 支持解析和序列化的文件格式：RDF/XML,N3,NTriples,N-Quads,Turtle,TriX,RDFa,Microdata,JSON-LD。
from rdflib import Graph
import random
import numpy as np
import tensorflow as tf
import math
import datetime as dt

# cPickle可以对任意一种类型的python对象进行序列化操作
# cPickle是python2的库，到python3，改名为 pickle
# import cPickle
import pickle

# kitchen includes functions to make gettext easier to use, handling unicode text easier
# (conversion with bytes, outputting xml, and calculating how many columns a string takes)
from kitchen.text.converters import getwriter, to_bytes, to_unicode
# i18n => 国际化：https://pythonhosted.org/kitchen/api-i18n.html
from kitchen.i18n import get_translation_object
translations = get_translation_object('example')
_ = translations.ugettext
b_ = translations.lgettext

import platform
print('Python version: %s' % platform.python_version())
print('Tensorflow version: %s' % tf.__version__)

# ======================== 1. 数据集加载：训练集/测试集 =================================

# Combine two KG
# 加载数据集
# DWY-NB consists of two datasets DY-NB and DW-NB; each dataset consists of a pair of KGs that can be used for the evaluation of EA techniques.
# The two KGs of DY-NB are subsets of DBpedia [Auer et al., 2007] and Yago [Hoffart et al., 2013], respectively.
# The two KGs of DW-NB are subsets of DBpedia and Wikidata [Vrandecic and Krotzsch, 2014].

# ttl是Turtle 格式的简称, 是RDF数据的表达格式之一。RDF的最初的的格式是xml/rdf格式，但是过于繁琐，turtle则直观简单很多
# Turtle数据描述：
# @prefix entity: <http://www.wikidata.org/entity#>
# @prefix rdf-schema: <http://www.w3.org/2000/01/rdf-schema#>
# @prefix XMLSchema: <http://www.w3.org/2001/XMLSchema#>
# entity:Q1376298 rdf-schema:label "Europe"
# entity:Q5312467 entity:P569 "1821-09-26" ^^XMLSchema:date
# 其实wd.ttl使用的是ntriples存储

# DW-NB数据集处理
# lgd_filename = 'dataset/DWY-NB/DW-NB/wd.ttl'  # (The subset of Wikidata KG)
# dbp_filename = 'dataset/DWY-NB/DW-NB/dbp_wd.ttl' # (The subset of DBpedia KG)
# map_file = 'dataset/DWY-NB/DW-NB/mapping_wd.ttl' # (The known entity alignment as testing data)

# DY-NB数据集处理
# yago_filename = 'dataset/DWY-NB/DY-NB/yago.ttl'  # (The subset of yago KG)
# dbp_filename = 'dataset/DWY-NB/DY-NB/dbp_yago.ttl' # (The subset of DBpedia KG)
# map_file = 'dataset/DWY-NB/DY-NB/mapping_yago.ttl' # (The known entity alignment as testing data)

# 测试数据集处理
yago_filename = 'dataset/DWY-NB/yago.ttl'  # (The subset of yago KG)
dbp_filename = 'dataset/DWY-NB/dbp.ttl' # (The subset of DBpedia KG)
map_file = 'dataset/DWY-NB/mapping.ttl' # (The known entity alignment as testing data)


# 创建一个图谱
graph = Graph()
# 解析wd.ttl, dbp_wd.ttl, 文件格式为ntriples - 解析训练集
# graph.parse(location=lgd_filename, format='nt')
graph.parse(location=yago_filename, format='nt')
graph.parse(location=dbp_filename, format='nt')

# 解析mapping_wd.ttl - 解析测试集
map_graph = Graph()
map_graph.parse(location=map_file, format='nt')


# ======================== 2. 存储实体标签集合(存储实体名称集合)： entity_label_dict =================================
# 实体标签label - 约等于实体名称.
# TODO 例如：<http://dbpedia.org/resource/Ettore_Puricelli> <http://www.w3.org/2000/01/rdf-schema#label> "Ettore Puricelli " .
#  entity_label_dict存储的内容为：entity_label_dict[<http://dbpedia.org/resource/Ettore_Puricelli>] =  "Ettore Puricelli "
entity_label_dict = dict()

# 遍历所解析的图谱（训练集图谱）
# Python 3中，unicode已重命名为str。 所以(str)(p) => (unicode)(p)
for s,p,o in graph:
    # 谓词为http://www.w3.org/2000/01/rdf-schema#label, 遂存储实体标签集合
    if (str)(p) == u'http://www.w3.org/2000/01/rdf-schema#label':
        entity_label_dict[s] = (str)(o)


# ======================== 3. 统计三元组（属性/关系三元组都统计）头实体个数：num_subj_triple =================================
# TODO 统计三元组的头实体个数
num_subj_triple = dict()
for s,p,o in graph:
    if num_subj_triple.get(s) == None:
        num_subj_triple[s] = 1
    else:
        num_subj_triple[s] += 1

# ======================== 3. 存储两个知识图谱的相交谓词（即同样的谓词） =================================
# 论文所给出的训练数据，已经预先进行了谓词对齐，将yago的相似谓词替换为dbpedia的谓词
# 但是也可能存在yago有, 但是dbpedia没有的谓词, 例如<http://yago-knowledge.org/resource/hasWikipediaArticleLength>
# 虽然谓词都一样, 不需要计算谓词相似性, 但是还是需要进行谓词替换, 以便落在同一个向量空间

### Automatically extracted intersection predicates ###
# 自动提取相交谓词
# intersection_predicates = ['http://www.wikidata.org/entity/P36',\
# 'http://www.wikidata.org/entity/P185',\
# 'http://www.wikidata.org/entity/P345',\
# 'http://www.wikidata.org/entity/P214',\
# 'http://www.wikidata.org/entity/P40',\
# 'http://www.wikidata.org/entity/P569',\
# 'http://www.wikidata.org/entity/P102',\
# 'http://www.wikidata.org/entity/P175',\
# 'http://www.wikidata.org/entity/P131',\
# 'http://www.wikidata.org/entity/P577',\
# 'http://www.wikidata.org/entity/P140',\
# 'http://www.wikidata.org/entity/P400',\
# 'http://www.wikidata.org/entity/P736',\
# 'http://www.wikidata.org/entity/P1432',\
# 'http://www.wikidata.org/entity/P159',\
# 'http://www.wikidata.org/entity/P136',\
# 'http://www.wikidata.org/entity/P1477',\
# 'http://www.wikidata.org/entity/P227',\
# 'http://www.wikidata.org/entity/P6',\
# 'http://www.wikidata.org/entity/P108',\
# 'http://www.wikidata.org/entity/P585',\
# 'http://www.wikidata.org/entity/P239',\
# 'http://www.wikidata.org/entity/P98',\
# 'http://www.wikidata.org/entity/P54',\
# 'http://www.wikidata.org/entity/P17',\
# 'http://www.wikidata.org/entity/P244',\
# 'http://www.wikidata.org/entity/P238',\
# 'http://www.wikidata.org/entity/P287',\
# 'http://www.wikidata.org/entity/P570',\
# 'http://www.wikidata.org/entity/P176',\
# 'http://www.wikidata.org/entity/P119',\
# 'http://www.wikidata.org/entity/P230',\
# 'http://www.wikidata.org/entity/P50',\
# 'http://www.wikidata.org/entity/P57',\
# 'http://www.wikidata.org/entity/P969',\
# 'http://www.wikidata.org/entity/P20',\
# 'http://www.wikidata.org/entity/P374',\
# 'http://www.wikidata.org/entity/P19',\
# 'http://www.wikidata.org/entity/P84',\
# 'http://www.wikidata.org/entity/P166',\
# 'http://www.wikidata.org/entity/P571',\
# 'http://www.wikidata.org/entity/P184',\
# 'http://www.wikidata.org/entity/P473',\
# 'http://www.wikidata.org/entity/P219',\
# 'http://www.wikidata.org/entity/P170',\
# 'http://www.wikidata.org/entity/P26',\
# 'http://www.wikidata.org/entity/P580',\
# 'http://www.wikidata.org/entity/P1015',\
# 'http://www.wikidata.org/entity/P408',\
# 'http://www.wikidata.org/entity/P172',\
# 'http://www.wikidata.org/entity/P220',\
# 'http://www.wikidata.org/entity/P177',\
# 'http://www.wikidata.org/entity/P178',\
# 'http://www.wikidata.org/entity/P161',\
# 'http://www.wikidata.org/entity/P27',\
# 'http://www.wikidata.org/entity/P742',\
# 'http://www.wikidata.org/entity/P607',\
# 'http://www.wikidata.org/entity/P286',\
# 'http://www.wikidata.org/entity/P361',\
# 'http://www.wikidata.org/entity/P1082',\
# 'http://www.wikidata.org/entity/P344',\
# 'http://www.wikidata.org/entity/P106',\
# 'http://www.wikidata.org/entity/P112',\
# 'http://www.wikidata.org/entity/P1036',\
# 'http://www.wikidata.org/entity/P229',\
# 'http://www.w3.org/2000/01/rdf-schema#label',\
# 'http://www.wikidata.org/entity/P126',\
# 'http://www.wikidata.org/entity/P750',\
# 'http://www.wikidata.org/entity/P144',\
# 'http://www.wikidata.org/entity/P69',\
# 'http://www.wikidata.org/entity/P264',\
# 'http://www.wikidata.org/entity/P218',\
# 'http://www.wikidata.org/entity/P110',\
# 'http://www.wikidata.org/entity/P86',\
# 'http://www.wikidata.org/entity/P957',\
# 'http://www.wikidata.org/entity/P1040',\
# 'http://www.wikidata.org/entity/P200',\
# 'http://www.wikidata.org/entity/P605',\
# 'http://www.wikidata.org/entity/P118',\
# 'http://www.wikidata.org/entity/P127']

# DY-NB的相交谓词
intersection_predicates = ['http://dbpedia.org/ontology/Person/height',
'http://dbpedia.org/ontology/PopulatedPlace/populationDensity',
'http://dbpedia.org/ontology/PopulatedPlace/populationMetroDensity',
'http://dbpedia.org/ontology/activeYearsEndDate',
'http://dbpedia.org/ontology/activeYearsStartYear',
'http://dbpedia.org/ontology/alias',
'http://dbpedia.org/ontology/almaMater',
'http://dbpedia.org/ontology/area',
'http://dbpedia.org/ontology/areaUrban',
'http://dbpedia.org/ontology/award',
'http://dbpedia.org/ontology/birthDate',
'http://dbpedia.org/ontology/birthName',
'http://dbpedia.org/ontology/birthPlace',
'http://dbpedia.org/ontology/birthYear',
'http://dbpedia.org/ontology/capital',
'http://dbpedia.org/ontology/child',
'http://dbpedia.org/ontology/citizenship',
'http://dbpedia.org/ontology/councilArea',
'http://dbpedia.org/ontology/country',
'http://dbpedia.org/ontology/county',
'http://dbpedia.org/ontology/currency',
'http://dbpedia.org/ontology/deathDate',
'http://dbpedia.org/ontology/deathPlace',
'http://dbpedia.org/ontology/deathYear',
'http://dbpedia.org/ontology/dissolutionYear',
'http://dbpedia.org/ontology/doctoralAdvisor',
'http://dbpedia.org/ontology/doctoralStudent',
'http://dbpedia.org/ontology/education',
'http://dbpedia.org/ontology/ethnicGroup',
'http://dbpedia.org/ontology/field',
'http://dbpedia.org/ontology/formerTeam',
'http://dbpedia.org/ontology/foundingYear',
'http://dbpedia.org/ontology/height',
'http://dbpedia.org/ontology/hometown',
'http://dbpedia.org/ontology/influenced',
'http://dbpedia.org/ontology/influencedBy',
'http://dbpedia.org/ontology/isPartOf',
'http://dbpedia.org/ontology/knownFor',
'http://dbpedia.org/ontology/language',
'http://dbpedia.org/ontology/largestCity',
'http://dbpedia.org/ontology/leader',
'http://dbpedia.org/ontology/length',
'http://dbpedia.org/ontology/lieutenancyArea',
'http://dbpedia.org/ontology/longName',
'http://dbpedia.org/ontology/mainInterest',
'http://dbpedia.org/ontology/managerClub',
'http://dbpedia.org/ontology/nationality',
'http://dbpedia.org/ontology/notableIdea',
'http://dbpedia.org/ontology/notableStudent',
'http://dbpedia.org/ontology/notableWork',
'http://dbpedia.org/ontology/officialLanguage',
'http://dbpedia.org/ontology/openingYear',
'http://dbpedia.org/ontology/party',
'http://dbpedia.org/ontology/picture',
'http://dbpedia.org/ontology/populationMetroDensity',
'http://dbpedia.org/ontology/populationTotal',
'http://dbpedia.org/ontology/principalArea',
'http://dbpedia.org/ontology/pseudonym',
'http://dbpedia.org/ontology/recordLabel',
'http://dbpedia.org/ontology/residence',
'http://dbpedia.org/ontology/restingPlace',
'http://dbpedia.org/ontology/spouse',
'http://dbpedia.org/ontology/stateOfOrigin',
'http://dbpedia.org/ontology/successor',
'http://dbpedia.org/ontology/team',
'http://dbpedia.org/ontology/weight',
'http://www.w3.org/2000/01/rdf-schema#label',
'http://www.w3.org/2003/01/geo/wgs84_pos#lat',
'http://www.w3.org/2003/01/geo/wgs84_pos#long',
'http://www.w3.org/2004/02/skos/core#prefLabel',
'http://xmlns.com/foaf/0.1/gender',
'http://xmlns.com/foaf/0.1/givenName',
'http://xmlns.com/foaf/0.1/surname']

# 相交谓词赋值
intersection_predicates_uri = intersection_predicates


# 定义若干数据处理方法. dataType: 返回数据类型；getRDFData：返回数据与数据类型；invert_dict：反转dict：并交换key,value
import rdflib
# 正则表达式
import re
import collections

# 字符量长度：保证向量长度统一
literal_len = 10

# TODO 返回字符类型: bit, int, float, text, string => todo：若是中文，需要进行额外处理(若不进行处理, 中文也会被定义为String)
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

# TODO Return: data, data_type 返回data,以及dataType
# o为rdflib类型, 它会解析ttl文件生成对应的三元组对象
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

# TODO 反转dict：并交换key,value值
# iteritems返回迭代器
# Python3 将iteritems()替换为items()
def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])

# o = getRDFData(o), 其中o[data, dataType], o[0]: 具体数据; o[1]: 数据类型
# literal_len = 10
# char_vocab: 字符向量

# Todo：literal_len = 10, 设定字面量的向量长度为10, 保证向量的统一, 长度更长的时候进行截断
# TODO: getLiteralArray作用: 为每个字符串生成向量表示, 即生成单一向量, 并且是独热编码, 从向量字典中获取单个字符的向量表示!
#  每个字符的向量为字符向量为len(char_vocab), char_vocab为向量字典.
#  literal_object：实体名称的向量表示/属性值的向量表示
# TODO: 改进： 字符串的向量化使用嵌入的方式？？ Word2vec？？ 而不是独热编码的方式.

def getLiteralArray(o, literal_len, char_vocab):
    literal_object = list()
    # literal_object初始化为0
    for i in range(literal_len):
        literal_object.append(0)

    # 判断数据类型是否是'uri',
    # 是字面量, 则对字面量进行处理 => 也可以看成对属性值的处理！！！
    if o[1] != 'uri':
        max_len = min(literal_len, len(o[0]))
        for i in range(max_len):
            # char_vocab没有字符o[0][i]对应的字符向量
            if char_vocab.get(o[0][i]) == None:
                char_vocab[o[0][i]] = len(char_vocab) # 字符向量为len(char_vocab)
            literal_object[i] = char_vocab[o[0][i]]

    # 是'uri', 并且entity_label_dict不为空, 对实体名称进行向量化
    # entity_label_dict存储的内容为：entity_label_dict[<http://dbpedia.org/resource/Ettore_Puricelli>] =  "Ettore Puricelli "
    elif entity_label_dict.get(o[0]) != None:
        # label形如："Ettore Puricelli "
        label = entity_label_dict.get(o[0])
        max_len = min(literal_len, len(label))
        for i in range(max_len):
            # char_vocab没有字符label[i]对应的字符向量
            if char_vocab.get(label[i]) == None:
                char_vocab[label[i]] = len(char_vocab)
            literal_object[i] = char_vocab[label[i]]

    return literal_object

# 头尾实体向量字典, 若尾巴实体是字面量, 则聚焦于其数据类型
# TODO 对于(s,p,o), 当遇到属性三元组时, 聚焦于属性的数据类型(会进行数据类型去重);
#  当遇到关系三元组时, 将关系三元组的头尾实体进行保存
entity_vocab = dict()

# dbp 头实体向量列表
# TODO 只保存dbp 头实体向量, 不考虑其谓词和尾实体
entity_dbp_vocab = list()

# 头尾实体负样本列表, 若尾实体是字面量, 则不会保存
# TODO 简单来说, 就是设置每个KG的负样本字面量, 排除尾实体为属性值的负样本
entity_dbp_vocab_neg = list()
entity_lgd_vocab_neg = list()

# 谓词向量字典
# TODO 简单来说, 就是将(s,p,o)中的p提取出来了, 并为每个p定义一个向量
predicate_vocab = dict()
predicate_vocab['<NONE>'] = 0

# 头尾实体字面量向量字典, 不管是URI, 还是字面量都会保存
# TODO 简单来说, 就是将(s,p,o)中的s,o提取出来了
# 例如entity_literal_vocab[<http://yago-knowledge.org/resource/Simon_Colosimo>] = len(entity_literal_vocab)
entity_literal_vocab = dict()

# 头尾实体负样本字面量列表, 不管是URI, 还是字面量都会保存
# TODO 简单来说, 就是设置每个KG的负样本字面量, 不会排除o为属性值的负样本
#  len(entity_literal_dbp_vocab_neg) + len(entity_literal_lgd_vocab_neg) = len(entity_literal_vocab)
entity_literal_dbp_vocab_neg = list()
entity_literal_lgd_vocab_neg = list()

#  TODO 存储谓词在相交谓词集合的关系三元组向量 [[[s,p,o,p_trans],[chars],predicate_weight], ... ]
data_uri = []

# TODO 存储谓词不在相交谓词集合的关系三元组向量 [[[s,p,属性数据类型,p_trans],[chars],predicate_weight], ... ]
data_uri_0 = []

# TODO 存储谓词不在相交谓词集合的属性三元组向量 [[[s,p,属性数据类型,p_trans],[chars],predicate_weight], ... ]
data_literal_0 = []

# TODO 存储谓词在相交谓词集合的属性三元组向量 [[[s,p,o,p_trans],[chars],predicate_weight], ... ]
data_literal = []

# TODO：我猜是按照传递规则所设置得到的新三元组！！ 丰富属性三元组和关系三原则：Transitivity Rule
# TODO 根据传递规则得到的关系三元组
# TODO 根据传递规则得到的属性三元组
data_uri_trans = []
data_literal_trans = []

# 字符向量字典
char_vocab = dict()
char_vocab['<pad>'] = 0

# 统计关系谓词/属性谓词个数 => 计算权重
pred_weight = dict()

# 三元组(s,p,o)个数
num_triples = 0

# TODO graph中的每一个三元组, 分别对每个三元组进行处理
for s, p, o in graph:
    # 统计三元组个数
    num_triples += 1

    # s/p/o = [data, data_type] => data_type指示了数据类型, 是uri, 还是字面量类型
    s = getRDFData(s)
    p = getRDFData(p)
    o = getRDFData(o)

    # 统计关系谓词/属性谓词个数
    if pred_weight.get(p[0]) == None:
        pred_weight[p[0]] = 1
    else:
        pred_weight[p[0]] += 1

    # 5. 设置所有头尾实体字面量向量字典（不论是关系三元组的头尾实体, 还是属性三元组的头尾实体都进行保存, 保存在entity_literal_vocab）
    #    和保存对应的头尾实体负样本字面量（保存在entity_literal_dbp_vocab_neg, entity_literal_lgd_vocab_neg）

    # 设置头实体s的字面量向量
    if entity_literal_vocab.get(s[0]) == None:
        # 设置头实体字面量向量：向量为len(entity_literal_vocab)
        entity_literal_vocab[s[0]] = len(entity_literal_vocab)
        # 并保存头实体负样本
        if (str)(s[0]).startswith(u'http://dbpedia.org/resource/'):
            entity_literal_dbp_vocab_neg.append(s[0])
        else:
            entity_literal_lgd_vocab_neg.append(s[0])

    # 设置尾实体o的字面量向量
    if entity_literal_vocab.get(o[0]) == None:
         # 设置尾实体字面量向量：向量为len(entity_literal_vocab)
        entity_literal_vocab[o[0]] = len(entity_literal_vocab)
        # 并保存尾实体负样本
        if (str)(s[0]).startswith(u'http://dbpedia.org/resource/'):
            entity_literal_dbp_vocab_neg.append(o[0])
        else:
            entity_literal_lgd_vocab_neg.append(o[0])

    # 6. 设置头实体、谓词、尾实体的实体向量字典（若尾实体是字面量, 则考虑其数据类型的向量（这也是和entity_literal_vocab的不同之处）,
    #  头尾实体保存在entity_vocab, 谓词保存在predicate_vocab）
    #  dbp的实体向量（保存在entity_dbp_vocab）
    #  和保存对应的头尾实体负样本, 若尾实体是字面量, 则不会保存（保存在entity_dbp_vocab_neg, entity_lgd_vocab_neg）

    # 设置头实体s向量
    if entity_vocab.get(s[0]) == None:
        idx = len(entity_vocab)
        entity_vocab[s[0]] = idx # 实体词向量为len(entity_vocab)
        if (str)(s[0]).startswith(u'http://dbpedia.org/resource/'):
            entity_dbp_vocab.append(idx)
            entity_dbp_vocab_neg.append(s[0])
        else:
            entity_lgd_vocab_neg.append(s[0])

    # 设置谓词向量字典 => len(predicate_vocab)
    if predicate_vocab.get(p[0]) == None:
        predicate_vocab[p[0]] = len(predicate_vocab)

    # 数据类型为'uri'
    if o[1] == 'uri':
        if entity_vocab.get(o[0]) == None:
            # 即尾实体还是一个关系, 设置尾实体的向量, 保存在头尾实体向量字典中
            entity_vocab[o[0]] = len(entity_vocab)
            if (str)(s[0]).startswith(u'http://dbpedia.org/resource/'):
                entity_dbp_vocab_neg.append(o[0])
            else:
                entity_lgd_vocab_neg.append(o[0])

        # 6. 根据谓词是否相交对关系三元组进行不同的处理, 不相交则存于data_uri_0, 相交存于data_uri
        #  TODO 返回尾实体的字面量数组, 为全0
        literal_object = getLiteralArray(o, literal_len, char_vocab)

        # 若当前谓词不在相交谓词中, 则将对应的实体向量进行存储data_uri_0
        # 具体实体向量通过向量字典得到
        if (str)(p[0]) not in intersection_predicates_uri:
            data_uri_0.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o[0]], 0], literal_object])
        else:
             # 若当前谓词在相交谓词中, 则将对应的头尾实体向量和谓词向量进行存储data_uri
            data_uri.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o[0]], 0], literal_object])

            ### DATA TRANS => 数据转换
            # TODO 丰富属性三元组和关系三原则：Transitivity Rule

            # 找到o[0]尾实体对应谓词的重复次数
            # Counter统计字符重复次数
            # A generator of predicates with the given subject and object
            duplicate_preds = [item for item, count in collections.Counter(graph.predicates(o[0],None)).items() if count > 1]
            if True:
              # 找到以o[0]实体开头的三元组, 进而根据传递规则丰富三元组数量
                for g1 in graph.triples((o[0],None,None)):
                    if len(g1) > 0:
                        s1,p1,o1 = g1

                        s1 = getRDFData(s1)
                        p1 = getRDFData(p1)
                        o1 = getRDFData(o1)

                        # 找到的额外三元组, 若没有设置对应的实体向量字典, 则进行设置, 以及保存对应的头尾实体负样本
                        if entity_vocab.get(o1[0]) == None:
                            entity_vocab[o1[0]] = len(entity_vocab)
                        if (str)(s1[0]).startswith(u'http://dbpedia.org/resource/'):
                            entity_dbp_vocab_neg.append(o1[0])
                        else:
                            entity_lgd_vocab_neg.append(o1[0])

                        # 若o1为字面量（属性值）, 没有设置对应的实体向量字典, 则进行设置 => key为数据类型
                        if entity_vocab.get(o1[1]) == None:
                            entity_vocab[o1[1]] = len(entity_vocab)

                        # 若没有设置对应的谓词向量字典, 则进行设置
                        if predicate_vocab.get(p1[0]) == None:
                            predicate_vocab[p1[0]] = len(predicate_vocab)

                        # 遍历过程中发现额外三元组的两个谓词不相等, 但是s[0]的谓词与intersection_predicates_uri存在交集
                        if p[0] != p1[0] and len(set((str)(x) for x in (graph.predicates(s[0]))).intersection(set(intersection_predicates_uri))) != 0:
                            # 尾实体o1[0]为URIRef 并且谓词存在于intersection_predicates_ur, 存储转换内容到data_uri_trans
                            if isinstance(o1[0], rdflib.term.URIRef) and (str)(p1[0]) in intersection_predicates_uri:
                                data_uri_trans.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o1[0]], predicate_vocab[p1[0]]], getLiteralArray(o1, literal_len, char_vocab)])
                            # 头实体为Literal 并且谓词为'http://www.w3.org/2000/01/rdf-schema#label'
                            elif isinstance(o1[0], rdflib.term.Literal) and (str)(p1[0]) == u'http://www.w3.org/2000/01/rdf-schema#label':
                                data_literal_trans.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o1[1]], predicate_vocab[p1[0]]], getLiteralArray(o1, literal_len, char_vocab)])
                              #tmp_data.append((s[0], p[0], o[0], p1[0], o1[0]))
              ##############
    else:
        # 数据类型为字面量, 进行字面量的处理
        if entity_vocab.get(o[1]) == None:
            entity_vocab[o[1]] = len(entity_vocab)
        # 对字面量进行向量化, 也可以看做对属性值进行向量化
        literal_object = getLiteralArray(o, literal_len, char_vocab)
        # 若当前谓词不在相交谓词中, 则将对应的三元组实体向量进行存储data_literal_0
        if (str)(p[0]) not in intersection_predicates:
            data_literal_0.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o[1]], 0], literal_object])
        else:
            # 若当前谓词在相交谓词中, 则将对应的三元组实体向量进行存储data_literal
            data_literal.append([[entity_vocab[s[0]], predicate_vocab[p[0]], entity_vocab[o[1]], 0], literal_object])

# 向量字典量翻转
reverse_entity_vocab = invert_dict(entity_vocab)
reverse_predicate_vocab = invert_dict(predicate_vocab)
reverse_char_vocab = invert_dict(char_vocab)
reverse_entity_literal_vocab = invert_dict(entity_literal_vocab)

#Add predicate weight => 增加谓词权重
for i in range(0, len(data_uri)):
    # data_uri = [] ###[ [[s,p,o,p_trans],[chars],predicate_weight], ... ]
    s = reverse_entity_vocab.get(data_uri[i][0][0])
    p = reverse_predicate_vocab.get(data_uri[i][0][1])
    # 谓词权重 = 出现次数 / 三元组总数
    data_uri[i].append([(pred_weight.get(p)/float(num_triples))])

# 下面的转换过程同上
for i in range(0, len(data_uri_0)):
    s = reverse_entity_vocab.get(data_uri_0[i][0][0])
    p = reverse_predicate_vocab.get(data_uri_0[i][0][1])
    data_uri_0[i].append([(pred_weight.get(p)/float(num_triples))])

for i in range(0, len(data_uri_trans)):
    s = reverse_entity_vocab.get(data_uri_trans[i][0][0])
    p = reverse_predicate_vocab.get(data_uri_trans[i][0][1])
    data_uri_trans[i].append([(pred_weight.get(p)/float(num_triples))])

for i in range(0, len(data_literal)):
    s = reverse_entity_vocab.get(data_literal[i][0][0])
    p = reverse_predicate_vocab.get(data_literal[i][0][1])
    data_literal[i].append([(pred_weight.get(p)/float(num_triples))])

for i in range(0, len(data_literal_0)):
    s = reverse_entity_vocab.get(data_literal_0[i][0][0])
    p = reverse_predicate_vocab.get(data_literal_0[i][0][1])
    data_literal_0[i].append([(pred_weight.get(p)/float(num_triples))])

for i in range(0, len(data_literal_trans)):
    s = reverse_entity_vocab.get(data_literal_trans[i][0][0])
    p = reverse_predicate_vocab.get(data_literal_trans[i][0][1])
    data_literal_trans[i].append([(pred_weight.get(p)/float(num_triples))])

#根据传递规则得到的新三元组小于100, 则进行double叠加
if len(data_uri_trans) < 100:
    data_uri_trans = data_uri_trans+data_uri_trans

print (len(entity_vocab), len(predicate_vocab), len(char_vocab), len(entity_dbp_vocab))



# 对上述向量进行持久化操作 - pickle模块实现了基本的数据序列化和反序列化。
# 通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储
pickle.dump(entity_literal_vocab, open("data/vocab_all.pickle", "wb"))
pickle.dump(char_vocab, open("data/vocab_char.pickle", "wb"))
pickle.dump(entity_vocab, open("data/vocab_entity.pickle", "wb"))
pickle.dump(predicate_vocab, open("data/vocab_predicate.pickle", "wb"))
pickle.dump(entity_dbp_vocab, open("data/vocab_kb1.pickle", "wb"))
pickle.dump(entity_dbp_vocab_neg, open("data/vocab_kb1_neg.pickle", "wb"))
pickle.dump(entity_lgd_vocab_neg, open("data/vocab_kb2_neg.pickle", "wb"))
pickle.dump(entity_label_dict, open("data/entity_label.pickle", "wb"))
pickle.dump(entity_literal_dbp_vocab_neg, open("data/vocab_kb1_all_neg.pickle", "wb"))
pickle.dump(entity_literal_lgd_vocab_neg, open("data/vocab_kb2_all_neg.pickle", "wb"))
pickle.dump(data_uri, open("data/data_uri.pickle", "wb"))
pickle.dump(data_uri_0, open("data/data_uri_n.pickle", "wb"))
pickle.dump(data_literal, open("data/data_literal.pickle", "wb"))
pickle.dump(data_literal_0, open("data/data_literal_n.pickle", "wb"))
pickle.dump(data_uri_trans, open("data/data_trans.pickle", "wb"))



# 查看pickle文件, 看看每个文件到底有什么东西
if __name__ == '__main__':

    # TODO entity_literal_vocab => 头尾实体rdflib.term.URIRef和rdflib.term.Literal, 以及对应的向量编号
    #  简单来说, 就是将(s,p,o)中的s,o提取出来了
    entity_literal_vocab = pickle.load(open("data/vocab_all.pickle", "rb"))
    print(entity_literal_vocab)
    print()

    # TODO char_vocab =>  字符向量编号
    char_vocab = pickle.load(open("data/vocab_char.pickle", "rb"))
    print(char_vocab)
    print()

    # TODO entity_vocab => 头尾实体rdflib.term.URIRef, 若尾实体是属性, 则保存对应的数据类型以及对应的向量编号
    #  对于(s,p,o), 当遇到属性三元组时, 聚焦于属性的数据类型(会进行数据类型去重); 当遇到关系三元组时, 将关系三元组的头尾实体进行保存
    entity_vocab = pickle.load(open("data/vocab_entity.pickle", "rb"))
    print(entity_vocab)
    print()

    # TODO entity_vocab => 谓词rdflib.term.URIRef 以及对应的向量编号
    #  简单来说, 就是将(s,p,o)中的p提取出来了, 并为每个p定义一个向量
    predicate_vocab = pickle.load(open("data/vocab_predicate.pickle", "rb"))
    print(predicate_vocab)
    print()

    #  TODO 只保存dbp 头实体向量, 不考虑其谓词和尾实体
    entity_dbp_vocab = pickle.load(open("data/vocab_kb1.pickle", "rb"))
    print(entity_dbp_vocab)
    print()

    # TODO 简单来说, 就是设置每个KG的负样本字面量, 排除尾实体为属性值的负样本
    entity_dbp_vocab_neg = pickle.load(open("data/vocab_kb1_neg.pickle", "rb"))
    print(entity_dbp_vocab_neg)
    print()
    entity_lgd_vocab_neg = pickle.load(open("data/vocab_kb2_neg.pickle", "rb"))
    print(entity_lgd_vocab_neg)
    print()

    # TODO 实体标签label - 约等于实体名称集合.
    entity_label_dict = pickle.load(open("data/entity_label.pickle", "rb"))
    print(entity_label_dict)
    print()

    # TODO 简单来说, 就是设置每个KG的负样本字面量, 不会排除o为属性值的负样本
    #  len(entity_literal_dbp_vocab_neg) + len(entity_literal_lgd_vocab_neg) = len(entity_literal_vocab)
    entity_literal_dbp_vocab_neg = pickle.load(open("data/vocab_kb1_all_neg.pickle", "rb"))
    print(entity_literal_dbp_vocab_neg)
    print()
    entity_literal_lgd_vocab_neg = pickle.load(open("data/vocab_kb2_all_neg.pickle", "rb"))
    print(entity_literal_lgd_vocab_neg)
    print()

    #  TODO 存储谓词在相交谓词集合的关系三元组向量 [[[s,p,o,p_trans],[chars],predicate_weight], ... ]
    data_uri = pickle.load(open("data/data_uri.pickle", "rb"))
    print(data_uri)
    print()

    # TODO 存储谓词不在相交谓词集合的关系三元组向量 [[[s,p,属性数据类型,p_trans],[chars],predicate_weight], ... ]
    data_uri_0 = pickle.load(open("data/data_uri_n.pickle", "rb"))
    print(data_uri_0)
    print()

    # TODO 存储谓词在相交谓词集合的属性三元组向量 [[[s,p,o,p_trans],[chars],predicate_weight], ... ]
    data_literal = pickle.load(open("data/data_literal.pickle", "rb"))
    print(data_literal)
    print()

    # TODO 存储谓词不在相交谓词集合的属性三元组向量 [[[s,p,属性数据类型,p_trans],[chars],predicate_weight], ... ]
    data_literal_0 = pickle.load(open("data/data_literal_n.pickle", "rb"))
    print(data_literal_0)
    print()

    # TODO：我猜是按照传递规则所设置得到的新三元组！！ 丰富属性三元组和关系三原则：Transitivity Rule
    # TODO 根据传递规则得到的关系三元组
    # TODO 根据传递规则得到的属性三元组
    data_uri_trans = pickle.load(open("data/data_trans.pickle", "rb"))
    print(data_uri_trans)
    print()
