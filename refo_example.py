# 参考链接： https://pythonmana.com/2022/03/202203120510336321.html

from refo import finditer, Predicate, Plus
import re
import copy

# 定义一个Word类, 同时保存了文本和文本的词性
class Word(object):
    def __init__(self, token, pos):
        self.token = token
        self.pos = pos

class W(Predicate):
    # token == none, token=".*", 否则为token等于传参
    def __init__(self, token=".*", pos=".*"):
        # re.compile()生成的是正则对象，单独使用没有任何意义，需要和findall(),
        # search(), match(）搭配使用
        self.token = re.compile(token + "$")
        self.pos = re.compile(pos + "$")
        super(W, self).__init__(self.match)

    # 判断文本和词性是否同时满足
    def match(self, word):
        """ 采用正则表达式同时匹配对象（word）的字符（token）和词性（pos） """
        m1 = self.token.match(word.token)
        m2 = self.pos.match(word.pos)
        return m1 and m2

class Rule(object):
    def __init__(self, condition=None, action=None):
        assert condition and action
        self.condition = condition
        self.action = action

    def apply(self, sentence):
        # 首先进行匹配操作，然后将匹配结果传入对应的操作函数中
        # 函数 finditer() 返回所有匹配的一个迭代器
        # 判断sentence是否满足匹配规则condition
        for m in finditer(self.condition, sentence):
            # 得到匹配的具体位置
            i, j = m.span()
            # 执行所传过来具体的action
            self.action(sentence[i:j])

def capitalize_name(x):
    """ 将英文单词的首字母大写 """
    for nnp in x:
        orig = nnp.token
        nnp.token = nnp.token.capitalize()
        print('Capitalized: {} -> {}'.format(orig, nnp.token))

def feet_to_mt(x):
    """ 将英尺转换为米 """
    number, units = x
    orig_number_token = number.token
    orig_units_token = units.token
    mt = float(number.token) * 0.3048
    number.token = "{0:.2}".format(mt)
    units.token = "mt."
    units.pos = "UNITS_METERS"
    print("feet_to_mt: {} {} -> {} {}".format(
    orig_number_token, orig_units_token, number.token, units.token))

# 定义具体的匹配规则
rules = [
    # 规则的目的：将英尺转换为米
    Rule(condition=W(pos="NUMBER") + W(pos="UNITS_FEET"),action=feet_to_mt),
    # 规则的目的：将姓名的英文单词的首字母大写
    Rule(condition=Plus(W(token="[^A-Z].*", pos="NNP")),action=capitalize_name),
]

sentence = "My|PRP friend|NN john|NNP smith|NNP is|VBZ 2|NUMBER " +\
"feet|UNITS_FEET taller|JJR than|IN mary|NNP Jane|NNP"

# 将句子进行分词, 并词性标注
sentence = [Word(*x.split("|")) for x in sentence.split()]

# 对原句子进行深拷贝
original = copy.deepcopy(sentence)

# 依次执行匹配规则
for rule in rules:
    rule.apply(sentence)

print("From: " + " ".join((w.token for w in original)))
print("To: " + " ".join((w.token for w in sentence)))