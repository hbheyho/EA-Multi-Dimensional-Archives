#!/usr/bin/env python
# coding: utf-8

import numpy as np

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import datetime as dt
# python3: cPickle => pickle
import pickle
import rdflib
import re

# from tensorflow.contrib import rnn
# rnn = tf.nn.rnn_cell.BasicRNNCell

# from tensorflow.contrib import rnn
from tensorflow.python.ops import rnn

import os
DEVICE = "0"
os.environ["CUDA_VISIBLE_DEVICES"]=DEVICE


# #### Load Data

# 反转dict：并交换key,value值
def invert_dict(d):
    return dict([(v, k) for k, v in d.items()]) # python3: iteritems() => items()

# 从pickle文件中加载预先处理数据
entity_literal_vocab = pickle.load(open("data/vocab_all.pickle", "rb")) # KB1 & KB2 entity and literal vocab => KB1 & KB2实体和字面量向量
char_vocab = pickle.load(open("data/vocab_char.pickle", "rb")) # KB1 & KB2 character vocab => KB1& KB2 字符向量
entity_vocab = pickle.load(open("data/vocab_entity.pickle", "rb")) # KB1 & KB2 entity vocab => KB1 & KB2 实体向量
predicate_vocab = pickle.load(open("data/vocab_predicate.pickle", "rb")) # KB1 & KB2 predicate vocab => KB1 & KB2谓词向量
entity_kb1_vocab = pickle.load(open("data/vocab_kb1.pickle", "rb")) # KB1 entity vocab for filtering final result => KB1(dbp)的实体向量
entity_kb1_vocab_neg = pickle.load(open("data/vocab_kb1_neg.pickle", "rb")) # KB1 entity & literal vocab for negative sampling => KB1 实体和字面量的负样本
entity_kb2_vocab_neg = pickle.load(open("data/vocab_kb2_neg.pickle", "rb")) # KB2 entity & literal vocab for negative sampling => KB2 实体和字面量的负样本
entity_label_dict = pickle.load(open("data/entity_label.pickle", "rb")) # KB1 & KB2 entity label => KB1 & KB2 实体标签
entity_literal_kb1_vocab_neg = pickle.load(open("data/vocab_kb1_all_neg.pickle", "rb")) # KB1 entity & literal vocab => KB1 实体和字面量负样本词向量
entity_literal_kb2_vocab_neg = pickle.load(open("data/vocab_kb2_all_neg.pickle", "rb")) # KB1 entity & literal vocab => KB2 实体和字面量负样本词向量

# 进行向量翻转
reverse_entity_vocab = invert_dict(entity_vocab)
reverse_predicate_vocab = invert_dict(predicate_vocab)
reverse_char_vocab = invert_dict(char_vocab)
reverse_entity_literal_vocab = invert_dict(entity_literal_vocab)

#relationship triples & attribute triples
# 加载关系三元组和属性三元组
data_uri = pickle.load(open("data/data_uri.pickle", "rb"))
data_uri_n = pickle.load(open("data/data_uri_n.pickle", "rb"))
data_literal = pickle.load(open("data/data_literal.pickle", "rb"))
data_literal_n = pickle.load(open("data/data_literal_n.pickle", "rb"))
data_trans = pickle.load(open("data/data_trans.pickle", "rb"))

print(len(entity_vocab)) # 133503



# 统计各个KG实体个数
dbp_size = 0
yag_size = 0
# wd_size = 0
for e in entity_vocab:
    print(e)
    if "http://dbpedia.org/resource/" in e:
        dbp_size += 1
#     elif "http://www.wikidata.org/entity/" in e:
#         wd_size += 1
    elif "yago-knowledge.org/resource/" in e:
        yag_size += 1
print(dbp_size)
print(yag_size)
# print(wd_size)


# #### Methods for data processing


# 返回字符类型
def dataType(string):
    odp='string'
    patternBIT=re.compile('[01]')
    patternINT=re.compile('[0-9]+')
    patternFLOAT=re.compile('[0-9]+\.[0-9]+')
    patternTEXT=re.compile('[a-zA-Z0-9]+')
    if patternTEXT.match(string):
        odp= "string"
    if patternINT.match(string):
        odp= "integer"
    if patternFLOAT.match(string):
        odp= "float"
    return odp

### Return: data, data_type
# 返回data,以及dataType
def getRDFData(o):
    if isinstance(o, rdflib.term.URIRef):
        data_type = "uri"
    else:
        data_type = o.datatype
        if data_type == None:
            data_type = dataType(o)
        else:
            if "#" in o.datatype:
                data_type = o.datatype.split('#')[1].lower()
            else:
                data_type = dataType(o)
        if data_type == 'gmonthday' or data_type=='gyear':
            data_type = 'date'
        if data_type == 'positiveinteger' or data_type == 'int' or data_type == 'nonnegativeinteger':
            data_type = 'integer'
    return o, data_type

# 反转dict：并交换key,value值
def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])

# 根据实体得到实体中每个字符向量数组：literal_len = 10
# o = getRDFData(o), 返回[data, dataType].
# char_vocab: 字符向量
def getLiteralArray(o, literal_len, char_vocab):
    literal_object = list()
    for i in range(literal_len):
        literal_object.append(0)
    if o[1] != 'uri':
        max_len = min(literal_len, len(o[0]))
        for i in range(max_len):
            if char_vocab.get(o[0][i]) == None:
                char_vocab[o[0][i]] = len(char_vocab)
            literal_object[i] = char_vocab[o[0][i]]
    elif entity_label_dict.get(o[0]) != None:
        label = entity_label_dict.get(o[0])
        max_len = min(literal_len, len(label))
        for i in range(max_len):
            if char_vocab.get(label[i]) == None:
                char_vocab[label[i]] = len(char_vocab)
            literal_object[i] = char_vocab[label[i]]
    return literal_object

# 对于一个有2000个训练样本的数据集。将2000个样本分成大小为500的batch，
# 那么完成一个epoch（阶段）需要4个iteration
# 通过current，batchSize, data计算出batch, 并随机挑选中负样本, 即生成训练数据
# 参考资料： https://www.jianshu.com/p/71f31c105879
def getBatch(data, batchSize, current, entityVocab, literal_len, char_vocab):
    # 判断是否有下一次batch训练
    hasNext = current+batchSize < len(data)
    
    if (len(data) - current) < batchSize:
        current = current - (batchSize - (len(data) - current))
    # 通过切片得到待处理数据
    dataPos_all = data[current:current+batchSize]
    
    # 处理后的处理数据集
    dataPos = list()
    charPos = list()
    pred_weight_pos = list()
    dataNeg = list()
    charNeg = list()
    
    pred_weight_neg = list()
    for triples, chars, pred_weight in dataPos_all:
        # 三元组以及转换后的谓词
        s,p,o,p_trans = triples
        dataPos.append([s,p,o,p_trans])
        charPos.append(chars)
        pred_weight_pos.append(pred_weight)
        
        lr = round(random.random())
        
        if lr == 0:
            try:
                o_type = getRDFData(reverse_entity_vocab[o]) # 头实体的[data, dataType]
            except:
                o_type = 'not_uri'
            
            literal_array = []
            rerun = True
            while rerun or negElm[0] == (reverse_entity_vocab[o] and literal_array == chars):
                if o_type[1] == 'uri':
                    if str(s).startswith('http://dbpedia.org/resource/'):
                        # 生成KG1 实体负样本
                        negElm = entity_kb1_vocab_neg[random.randint(0, len(entity_kb1_vocab_neg)-1)]
                        negElm = reverse_entity_vocab[entity_vocab[negElm]]
                    else:
                        # 生成KG2 实体负样本
                        negElm = entity_kb2_vocab_neg[random.randint(0, len(entity_kb2_vocab_neg)-1)]
                        negElm = reverse_entity_vocab[entity_vocab[negElm]]
                else:
                    if str(s).startswith('http://dbpedia.org/resource/'):
                        # 生成KG1字面量负样本
                        negElm = entity_literal_kb1_vocab_neg[random.randint(0, len(entity_literal_kb1_vocab_neg)-1)]
                        negElm = reverse_entity_literal_vocab[entity_literal_vocab[negElm]]
                    else:
                        # 生成KG2字面量负样本
                        negElm = entity_literal_kb2_vocab_neg[random.randint(0, len(entity_literal_kb2_vocab_neg)-1)]
                        negElm = reverse_entity_literal_vocab[entity_literal_vocab[negElm]]
                # 负样本生成完成
                if o_type == 'uri' and negElm[1] == 'uri':
                    rerun = False
                elif o_type != 'uri':
                    rerun = False
                # 生成负样本的的字符向量
                if (isinstance(negElm, rdflib.term.URIRef)) or (isinstance(negElm, rdflib.term.Literal)):
                    negElm = getRDFData(negElm)
                    literal_array = getLiteralArray(negElm, literal_len, char_vocab)
                else:
                    rerun = True  
            # 添加负样本数据
            if negElm[1] == 'uri':
                dataNeg.append([s, p, entity_vocab[negElm[0]], p_trans])
            else:
                dataNeg.append([s, p, entity_vocab[negElm[1]], p_trans])
            # 保存负样本的字符向量
            charNeg.append(literal_array)
            # 保存负样本权重
            pred_weight_neg.append(pred_weight)
        else:
            negElm = random.randint(0, len(entity_vocab)-1)
            # 若负样本等于尾实体, 则继续进行挑选
            while negElm == s:
                negElm = random.randint(0, len(entity_vocab)-1)
            dataNeg.append([negElm, p, o, p_trans])
            charNeg.append(chars)
            pred_weight_neg.append(pred_weight)
    
    # 将上述结果进行数组转换
    dataPos = np.array(dataPos)
    charPos = np.array(charPos)
    pred_weight_pos = np.array(pred_weight_pos)
    dataNeg = np.array(dataNeg)
    charNeg = np.array(charNeg)
    pred_weight_neg = np.array(pred_weight_neg)
    return hasNext, current+batchSize, dataPos[:,0], dataPos[:,1], dataPos[:,2], dataPos[:,3], pred_weight_pos, charPos, dataNeg[:,0], dataNeg[:,1], dataNeg[:,2], dataNeg[:,3], pred_weight_neg, charNeg 


# #### Hyperparameter


batchSize = 100  # batchSize长度
hidden_size = 100 # 隐藏层数
totalEpoch = 50  # 训练次数
verbose = 1000 # ??
margin = 1.0 # ??
literal_len = 10 # 字符量长度
entitySize = len(entity_vocab) # 实体长度
predSize = len(predicate_vocab) # 谓词长度
charSize = len(char_vocab) # 字符长度
top_k = 10 # top_k


# #### Prepare testing data


import random
from rdflib import URIRef

# 测试数据集
# file_mapping = open("data/mapping_wd.ttl", 'r')
# file_mapping = open("data/mapping_yago.ttl", 'r')
file_mapping = open("data/mapping.ttl", 'r')

test_dataset_list = list()
for line in file_mapping:
    elements = line.split(' ')
    s = elements[0]
    p = elements[1]
    o = elements[2]
    # python3:默认使用的是UTF-8编码。
    s = s.encode('utf-8').decode('unicode_escape')
    o = o.encode('utf-8').decode('unicode_escape')
    # 判断测试数据集的头尾实体是否在entity_vocab中
    if (entity_vocab[URIRef(s.replace('<','').replace('>',''))] in entity_kb1_vocab) and (URIRef(o.replace('<','').replace('>','')) in entity_vocab):
        test_dataset_list.append((o, s))
file_mapping.close()

# 确定测试集合的entity向量
# test_input => 头实体向量
# test_answer => 尾实体向量
test_input = [entity_vocab[URIRef(k.replace('<','').replace('>',''))] for k,_ in test_dataset_list]
test_answer = [entity_kb1_vocab.index(entity_vocab[URIRef(k.replace('<','').replace('>',''))]) for _,k in test_dataset_list]


# #### Embedding model


# 嵌入学习
tfgraph = tf.Graph()

with tfgraph.as_default():
    # 构建tensorflow流图的占位符合：
    # dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
    # shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
    # name：名称
    # h + r ≈ t
    pos_h = tf.placeholder(tf.int32, [None])
    pos_t = tf.placeholder(tf.int32, [None])
    pos_r = tf.placeholder(tf.int32, [None])
    pos_r_trans = tf.placeholder(tf.int32, [None])
    pos_c = tf.placeholder(tf.int32, [None, literal_len])
    pos_pred_weight = tf.placeholder(tf.float32, [None,1], name='pos_pred_weight')

    neg_h = tf.placeholder(tf.int32, [None])
    neg_t = tf.placeholder(tf.int32, [None])
    neg_r = tf.placeholder(tf.int32, [None])
    neg_r_trans = tf.placeholder(tf.int32, [None])
    neg_c = tf.placeholder(tf.int32, [None, literal_len])
    neg_pred_weight = tf.placeholder(tf.float32, [None,1], name='neg_pred_weight')
    
    type_data = tf.placeholder(tf.int32, [1])
    type_trans = tf.placeholder(tf.int32, [1])
    
    # 关系/实体嵌入；属性嵌入；关系嵌入；属性字符嵌入等等   ？？？？
    # 如果变量存在，函数tf.get_variable()会返回现有的变量；
    # 如果变量不存在，会根据给定形状和初始值创建一个新的变量
    # name：变量名称
    # shape：变量维度
    # initializer：变量初始化方式
    # regularizer：正规化
    # caching_device：可选的设备字符串或函数描述
    # tensorflow: tf.contrib.layers.xavier_initializer => tf.keras.initializers.glorot_normal()
    ent_embeddings_ori = tf.get_variable(name = "relationship_ent_embedding", shape = [entitySize, hidden_size], initializer = tf.keras.initializers.glorot_normal())
    atr_embeddings_ori = tf.get_variable(name = "attribute_ent_embedding", shape = [entitySize, hidden_size], initializer = tf.keras.initializers.glorot_normal())
    rel_embeddings = tf.get_variable(name = "rel_embedding", shape = [predSize, hidden_size], initializer = tf.keras.initializers.glorot_normal())
    attribute_rel_embeddings = tf.get_variable(name = "attribute_rel_embedding", shape = [predSize, hidden_size], initializer = tf.keras.initializers.glorot_normal())
    char_embeddings = tf.get_variable(name = "attribute_char_embedding", shape = [charSize, hidden_size], initializer = tf.keras.initializers.glorot_normal())
    
    # concat: 向量拼接
    # axis=0:代表在第0个维度拼接
    # axis=1:代表在第1个维度拼接 
    ent_indices = tf.concat([pos_h, pos_t, neg_h, neg_t], 0)
    # reshape: 张量重塑
    # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # reshape(t, [3, 3]) ==> [[1, 2, 3],
    #                         [4, 5, 6],
    #                         [7, 8, 9]]
    ent_indices = tf.reshape(ent_indices,[-1,1])
    
    # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素
    # tf.nn.embedding_lookup（params, ids）:params可以是张量也可以是数组等，id就是对应的索引
    ent_value = tf.concat([tf.nn.embedding_lookup(ent_embeddings_ori, pos_h),                          tf.nn.embedding_lookup(ent_embeddings_ori, pos_t),                          tf.nn.embedding_lookup(ent_embeddings_ori, neg_h),                          tf.nn.embedding_lookup(ent_embeddings_ori, neg_t)], 0)
    
    # 把源数据中的元素根据标记的下标位置indice分散到新数组的位置中去
    # indice = tf.constant([[4], [3], [1], [7]])
    # updates = tf.constant([9, 10, 11, 12])
    # shape = tf.constant([8]) 
    # a = tf.scatter_nd(indice, updates, shape)  a = tf.Tensor([ 0 11  0 10  9  0  0 12], shape=(8,), dtype=int32)
    part_ent_embeddings = tf.scatter_nd([ent_indices], [ent_value], ent_embeddings_ori.shape)
    # tf.stop_gradient 来对流经网络某部分的梯度流进行限制.
    # 可能会有这样的场景, 即我们可能只需要训练网络的特定部分, 然后网络的其余部分则保持未之前的状态
    ent_embeddings = part_ent_embeddings + tf.stop_gradient(-part_ent_embeddings + ent_embeddings_ori)
    
    # 对属性嵌入也进行上述和实体嵌入同样的方法进行处理
    atr_indices = tf.concat([pos_h, pos_t, neg_h, neg_t], 0)
    atr_indices = tf.reshape(atr_indices,[-1,1])
    atr_value = tf.concat([tf.nn.embedding_lookup(atr_embeddings_ori, pos_h),                          tf.nn.embedding_lookup(atr_embeddings_ori, pos_t),                          tf.nn.embedding_lookup(atr_embeddings_ori, neg_h),                          tf.nn.embedding_lookup(atr_embeddings_ori, neg_t)], 0)
    part_atr_embeddings = tf.scatter_nd([atr_indices], [atr_value], atr_embeddings_ori.shape)
    atr_embeddings = part_atr_embeddings + tf.stop_gradient(-part_atr_embeddings + atr_embeddings_ori)
    
    # tf.cond()函数用于控制数据流向
    # # 用于有条件的执行函数，当pred为True时，执行true_fn函数，否则执行false_fn函数
    # tf.cond(
    #    pred,
    #    true_fn=None,
    #    false_fn=None,
    #    strict=False,
    #    name=None,
    #    fn1=None,
    #    fn2=None
    #)
    # lambda 函数是一种小的匿名函数。
    # lambda 函数可接受任意数量的参数，但只能有一个表达式. x = lambda a : a + 10
    pos_h_e = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(ent_embeddings, pos_h), lambda: tf.nn.embedding_lookup(atr_embeddings, pos_h))
    pos_t_e = tf.cond(type_data[0] > 0, lambda: tf.stop_gradient(tf.nn.embedding_lookup(ent_embeddings, pos_t)), lambda: tf.nn.embedding_lookup(atr_embeddings, pos_t))
    pos_r_e = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(rel_embeddings, pos_r), lambda: tf.nn.embedding_lookup(attribute_rel_embeddings, pos_r))
    pos_r_e_trans = tf.nn.embedding_lookup(rel_embeddings, pos_r_trans)
    pos_c_e = tf.nn.embedding_lookup(char_embeddings, pos_c)
    
    # 负样本处理
    neg_h_e = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(ent_embeddings, neg_h), lambda: tf.nn.embedding_lookup(atr_embeddings, neg_h))
    neg_t_e = tf.cond(type_data[0] > 0, lambda: tf.stop_gradient(tf.nn.embedding_lookup(ent_embeddings, neg_t)), lambda: tf.nn.embedding_lookup(atr_embeddings, neg_t))
    neg_r_e = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(rel_embeddings, neg_r), lambda: tf.nn.embedding_lookup(attribute_rel_embeddings, neg_r))
    neg_r_e_trans = tf.nn.embedding_lookup(rel_embeddings, neg_r_trans)
    neg_c_e = tf.nn.embedding_lookup(char_embeddings, neg_c)
    
    # 若存在谓词转换, 则进行额外处理
    pos_r_e = tf.cond(type_trans[0] < 1, lambda: pos_r_e, lambda: tf.multiply(pos_r_e, pos_r_e_trans))
    neg_r_e = tf.cond(type_trans[0] < 1, lambda: neg_r_e, lambda: tf.multiply(neg_r_e, neg_r_e_trans))
    
    # 返回来一个给定形状和类型的用0填充的数组；
    # zeros(shape, dtype=float, order=‘C’)
    # np.zeros((2,5))
    # 结果为一个2行5列的矩阵
    # [[0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0.]]
    # Mask 是相对于 PAD 而产生的技术，具备告诉模型一个向量有多长的功效
    # Mask 矩阵是与 PAD 之后的矩阵具有相同的 shape
    # mask 矩阵只有 1 和 0两个值，如果值为 1 表示 PAD 矩阵中该位置的值有意义，值为 0 则表示对应 PAD 矩阵中该位置的值无意义
    mask_constant_0 = np.zeros([1,hidden_size])
    mask_constant_1 = np.ones([1,hidden_size])
    # np.concatenate: 数组拼接
    mask_constant = np.concatenate([mask_constant_0, mask_constant_1])
    # tf.constant: 创建常量
    mask_constant = tf.constant(mask_constant, tf.float32)
    
    # tf.sign(x, name=None), 返回符号 y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x>0.
    # tf.abs()就是求数值的绝对值
    # flag_pos_c_e：字符向量标记
    flag_pos_c_e = tf.sign(tf.abs(pos_c))
    # 根据字符标记, 进行mask操作
    mask_pos_c_e = tf.nn.embedding_lookup(mask_constant, flag_pos_c_e)
    #  pos_c_e * mask_pos_c_e 处理
    pos_c_e = pos_c_e * mask_pos_c_e
    
    # 对负样本的字符进行同样处理
    flag_neg_c_e = tf.sign(tf.abs(neg_c))
    mask_neg_c_e = tf.nn.embedding_lookup(mask_constant, flag_neg_c_e)
    neg_c_e = neg_c_e * mask_neg_c_e
    
    # 计算n-gram模型权重
    def calculate_ngram_weight(unstacked_tensor):
        # tf.stack( values,axis=0,name=’stack’)：将两个数组按照指定的方向进行叠加，生成一个新的数组
        stacked_tensor = tf.stack(unstacked_tensor, 1)
        # tf.reverse(): 反序, 相当于矩阵的初等行列变换
        stacked_tensor = tf.reverse(stacked_tensor, [1])
        index = tf.constant(len(unstacked_tensor))
        expected_result = tf.zeros([batchSize, hidden_size])
        def condition(index, summation):
            return tf.greater(index, 0) # tf.greater: 比较a、b两个值的大小
        def body(index, summation):
            precessed = tf.slice(stacked_tensor,[0,index-1,0], [-1,-1,-1]) # tf.slice()函数的作用就是从张量中提取想要的切片
            # tf.reduce_mean函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值
            summand = tf.reduce_mean(precessed, 1)
            return tf.subtract(index, 1), tf.add(summation, summand) # subtract：减法; add: 加法
        # def while_loop(cond, ### 一个函数，负责判断循环是否进行
        #       body,          ### 一个函数，循环体，更新变量
        #       loop_vars,     ### 初始循环变量，可以是多个，这些变量是cond、body 的输入和输出
        #       shape_invariants=None,
        #       parallel_iterations=10,
        #       back_prop=True,
        #       swap_memory=False,
        #       name=None,
        #       maximum_iterations=None,
        #       return_same_structure=False):
        result = tf.while_loop(condition, body, [index, expected_result])
        return result[1]
    
    # 基于LSTN进行处理
    # tf.unstack：以指定的轴axis, 将一个维度为R的张量数组转变成一个维度为R-1的张量
    # 即将一组张量以指定的轴，减少一个维度。正好和stack()相反
    pos_c_e_in_lstm = tf.unstack(pos_c_e, literal_len, 1)
    pos_c_e_lstm = calculate_ngram_weight(pos_c_e_in_lstm)
    
    neg_c_e_in_lstm = tf.unstack(neg_c_e, literal_len, 1)
    neg_c_e_lstm = calculate_ngram_weight(neg_c_e_in_lstm)
    
    tail_pos = tf.cond(type_data[0] > 0, lambda: pos_t_e, lambda: pos_c_e_lstm)
    tail_neg = tf.cond(type_data[0] > 0, lambda: neg_t_e, lambda: neg_c_e_lstm)
    
    # 基于pos_h_e + pos_r_e - tail_pos：h + r ≈ t
    # reduce_sum() 用于计算张量tensor沿着某一维度的和
    pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - tail_pos), 1, keep_dims = True)
    neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - tail_neg), 1, keep_dims = True)
    
    # 判断是否额外进行谓词权重处理
    pos = tf.cond(type_data[0] > 0, lambda: pos, lambda: tf.multiply(pos, pos_pred_weight))
    neg = tf.cond(type_data[0] > 0, lambda: neg, lambda: tf.multiply(neg, neg_pred_weight))
    # 学习率
    learning_rate = tf.cond(type_data[0] > 0, lambda: 0.01, lambda: tf.reduce_min(pos_pred_weight)*0.01)
    
    # tensorflow变量： tf.trainable_variables () 指的是需要训练的变量；tf.all_variables() 指的是所有变量
    opt_vars_ent = [v for v in tf.trainable_variables() if v.name.startswith("relationship") or v.name.startswith("rel_embedding")]
    opt_vars_atr = [v for v in tf.trainable_variables() if v.name.startswith("attribute") or v.name.startswith("attribute_rel_embedding") or v.name.startswith("rnn")]
    opt_vars_sim = [v for v in tf.trainable_variables() if v.name.startswith("relationship_ent_embedding") or v.name.startswith("attribute_rel_embedding")]
    opt_vars = [v for v in tf.trainable_variables()]
    
    ent_emb = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(ent_embeddings, pos_t), lambda: tf.nn.embedding_lookup(ent_embeddings, pos_h))
    atr_emb = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(atr_embeddings, pos_t), lambda: tf.nn.embedding_lookup(atr_embeddings, pos_h))
    # tf.nn.l2_normalize：利用 L2 范数对指定维度 dim 进行标准化
    norm_ent_emb = tf.nn.l2_normalize(ent_emb,1)
    norm_atr_emb = tf.nn.l2_normalize(atr_emb,1)
    cos_sim = tf.reduce_sum(tf.multiply(norm_ent_emb, norm_atr_emb), 1, keep_dims=True)
    sim_loss = tf.reduce_sum(1-cos_sim)
    # Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正
    sim_optimizer = tf.train.AdamOptimizer(0.01).minimize(sim_loss, var_list=opt_vars_sim)
    
    # 损失函数
    loss = tf.cond(type_data[0] > 0, lambda: tf.reduce_sum(tf.maximum(pos - neg + 1, 0) + (1-cos_sim)), lambda: tf.reduce_sum(tf.maximum(pos - neg + 1, 0)))
    loss = tf.cond(type_trans[0] < 1, lambda: loss, lambda: tf.multiply(loss, 0.1))
    # 利用Adam进行优化
    optimizer = tf.cond(type_data[0] > 0, lambda: tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=opt_vars_ent), lambda: tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=opt_vars_atr))
    
    # f.square()是对a里的每一个元素求平方
    # f.sqrt()是对a里的每一个元素求平方根
    norm = tf.sqrt(tf.reduce_sum(tf.square(ent_embeddings_ori), 1, keep_dims=True))
    # 标准化嵌入
    normalized_embeddings = ent_embeddings_ori / norm
    
    # 测试数据集
    test_dataset = tf.constant(test_input, dtype=tf.int32)
    # 测试数据集嵌入
    test_embeddings = tf.nn.embedding_lookup(normalized_embeddings, test_dataset)
    # 将矩阵 a 乘以矩阵 b,生成a * b, 得到相似性
    similarity = tf.matmul(test_embeddings, normalized_embeddings, transpose_b=True)
    
    # 初始化模型的参数
    init = tf.global_variables_initializer()


from functools import reduce  # py3

# 评价标准测试
# Mean rank/hits1
def metric(y_true, y_pred, answer_vocab, k=10):
    list_rank = list()
    total_hits = 0
    total_hits_1 = 0
    for i in range(len(y_true)):
        result = y_pred[i]
        result = result[answer_vocab]
        # 将矩阵a按照axis排序，并返回排序后的下标
        result = (-result).argsort()
        
        for j in range(len(result)):
            if result[j] == y_true[i]:
                rank = j
                break
        list_rank.append(j)
        
        result = result[:k]
        for j in range(len(result)):
            if result[j] == y_true[i]:
                total_hits += 1
                if j == 0:
                    total_hits_1 += 1
                break    
    return reduce(lambda x, y: x + y, list_rank) / len(list_rank), float(total_hits)/len(y_true), float(total_hits_1)/len(y_true)


# 运行函数
def run(graph, totalEpoch):
    # writer = open('log.txt', 'w', 0)
    writer = open('log.txt', 'wb', 0)
    with tf.Session(graph=graph) as session:
        init.run()
        # 进行totalEpoch个运行
        for epoch in range(totalEpoch):
            if epoch % 2 == 0:
                data = [data_uri_n, data_uri, data_literal_n, data_literal,[], data_trans]
            else:
                data = [[],[],data_literal_n,data_literal,[],data_trans]
            # 当前epoch开始时间
            start_time_epoch = dt.datetime.now()
            for i in range(0, len(data)):
                # Python random.shuffle() 函数将序列中的元素随机打乱
                random.shuffle(data[i])
                hasNext = True
                current = 0
                step = 0
                # 平均损失
                average_loss = 0
                
                if i > 3:
                    transitive = 1
                else:
                    transitive = 0
                
                if i == 0 or i ==1 or i == 4:
                    uri = 1
                else:
                    uri = 0
                  
                while(hasNext and len(data[i]) > 0):
                    step += 1
                    # 通过getBatch进行数据训练处理
                    hasNext, current, ph, pr, pt, pr_trans, ppred, pc, nh, nr, nt, nr_trans, npred, nc = getBatch(data[i], batchSize, current, entity_vocab, literal_len, char_vocab)
                    # 等待训练的数据
                    feed_dict = {
                        pos_h: ph,
                        pos_t: pt,
                        pos_r: pr,
                        pos_r_trans: pr_trans,
                        pos_pred_weight : ppred,
                        pos_c: pc,
                        neg_h: nh,
                        neg_t: nt,
                        neg_r: nr,
                        neg_r_trans: nr_trans,
                        neg_c: nc,
                        neg_pred_weight: npred,
                        type_data : np.full([1],uri),
                        type_trans : np.full([1],transitive)
                    }
                    # 不同的epoch进行不同处理
                    if epoch % 2 == 0:
                        __, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                        average_loss += loss_val
                    else:
                        __, loss_val = session.run([sim_optimizer, sim_loss], feed_dict=feed_dict)
                        average_loss += loss_val

                    if step % verbose == 0:
                        average_loss /= verbose
                        print('Epoch: ', epoch, ' Average loss at step ', step, ': ', average_loss)
                        # 书写日志
                        writer.write(('Epoch: '+ str(epoch) + ' Average loss at step '+ str(step) + ': '+ str(average_loss) +'\n').encode())
                        average_loss = 0
                if len(data[i]) > 0:
                        average_loss /= ((len(data[i])%(verbose*batchSize))/batchSize)
                        print('Epoch: ', epoch, ' Average loss at step ', step, ': ', average_loss)
                        writer.write(('Epoch: '+ str(epoch) + ' Average loss at step '+ str(step) + ': '+ str(average_loss) + '\n').encode())

            end_time_epoch = dt.datetime.now()
            print("Training time took {} seconds to run 1 epoch".format((end_time_epoch-start_time_epoch).total_seconds()))
            writer.write(("Training time took {} seconds to run 1 epoch\n".format((end_time_epoch-start_time_epoch).total_seconds())).encode())
            # 每10个epoch就进行一次metric评估
            if (epoch+1) % 10 == 0:
                start_time_epoch = dt.datetime.now()
                sim = similarity.eval()
                #  MeanRank, hits@10, hits@1
                mean_rank, hits_at_10, hits_at_1 = metric(test_answer, sim, entity_kb1_vocab, top_k)
                print("Mean Rank: ", mean_rank, " of ", len(entity_kb1_vocab))
                writer.write(("Mean Rank: "+ str(mean_rank) + " of "+ str(len(entity_kb1_vocab)) + "\n").encode())
                print("Hits @ "+str(top_k)+": ", hits_at_10)
                writer.write(("Hits @ "+str(top_k) +": "+ str(hits_at_10) + "\n").encode())
                print("Hits @ "+str(1)+": ", hits_at_1)
                writer.write(("Hits @ "+str(1) +": "+ str(hits_at_1) + "\n").encode())
                end_time_epoch = dt.datetime.now()
                print("Testing time took {} seconds.".format((end_time_epoch-start_time_epoch).total_seconds()))
                writer.write(("Testing time took {} seconds.\n\n".format((end_time_epoch-start_time_epoch).total_seconds())).encode())
                print
    writer.close()


start_time = dt.datetime.now()
run(tfgraph, totalEpoch) 
end_time = dt.datetime.now()
print("Training time took {} seconds to run {} epoch".format((end_time-start_time).total_seconds(), totalEpoch))

if __name__ == '__main__':
    print(1)