import jieba.analyse
import codecs
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import re

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# 进行jieba分词操作
def wordSegmentation():
    source = codecs.open('D:/StudyFile/ProjectWorkstation/EA/EntityAlignment/dataset/Segment/in_the_name_of_people.txt', 'r',
                         encoding="utf8")
    target = codecs.open(
        'D:/StudyFile/ProjectWorkstation/EA/EntityAlignment/dataset/Segment/in_the_name_of_people_segment.txt', 'w',
        encoding="utf8")
    dict_path = 'D:/StudyFile/ProjectWorkstation/EA/EntityAlignment/dataset/Segment/people.txt'

    # 加载自定义词典
    jieba.load_userdict(dict_path)

    print('Open Files')
    line = source.readline()

    # 循环遍历每一行，并对这一行进行分词操作
    # 如果下一行没有内容的话，就会readline会返回-1，则while -1就会跳出循环
    while line:
        # 去除标点符号
        remove_chars = '[·’!。"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
        remove_chars_re = re.compile(remove_chars)
        # 去重标点符号并去除头尾空格
        line = remove_chars_re.sub(r"", line.strip(' '))

        # jieba分词默认有：全模式、精确模式、搜索引擎模式, 不同模式有着不同的效果.
        # 具体见: https://blog.csdn.net/weixin_30254435/article/details/101571045
        line_seg = " ".join(jieba.cut(line))

        target.writelines(line_seg)
        line = source.readline()

    # 关闭两个文件流，并退出程序
    source.close()
    target.close()
    print("分词完成")


# Word2Vec第一个参数代表要训练的语料
# sg=1 表示使用Skip-Gram模型进行训练
# size 表示特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
# window 表示当前词与预测词在一个句子中的最大距离是多少
# min_count 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
# workers 表示训练的并行数
# sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)

def word2vec():
    # 首先打开需要训练的文本
    # sentences = word2vec.LineSentence('D:/StudyFile/ProjectWorkstation/EA/EntityAlignment/dataset/Segment/in_the_name_of_people_segment.txt')
    sentences = open('D:/StudyFile/ProjectWorkstation/EA/EntityAlignment/dataset/Segment'
                     '/in_the_name_of_people_segment.txt', 'rb')
    # 通过Word2vec进行训练
    model = Word2Vec(LineSentence(sentences), sg=1, vector_size=100, window=10, min_count=5, workers=15, sample=1e-3)
    # 保存训练好的模型
    model.save('D:/StudyFile/ProjectWorkstation/EA/EntityAlignment/'
               'dataset/Segment/in_the_name_of_people.word2vec')

    print('训练完成')


if __name__ == '__main__':
    # wordSegmentation()
    word2vec()
    # jhekrqoplw;eqweqweqwe
