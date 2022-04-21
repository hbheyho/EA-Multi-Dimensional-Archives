from rdflib import Graph
import rdflib
lgd_filename = 'dataset/DW-NB/rdflib_example.ttl'  # (The subset of Wikidata KG)
# 创建一个图谱
graph = Graph()
# 解析wd.ttl, dbp_wd.ttl, 文件格式为ntriples
graph.parse(location=lgd_filename, format='nt')

# ntriples：<http://dbpedia.org/resource/I_Feel_Lucky> <http://www.wikidata.org/entity/P577> "1992-05-18"^^<http://www.w3.org/2001/XMLSchema#date> .
for s, p, o in graph:
    print("==========================================")
    print(s) # http://dbpedia.org/resource/Jay_Karnes
    print(p) # http://www.w3.org/2000/01/rdf-schema#label
    print(o) # Jay Karnes

    print("==========================================")
    print("s是否是URLRef类型：", isinstance(s, rdflib.term.URIRef)) # True
    print("p是否是URLRef类型：", isinstance(p, rdflib.term.URIRef)) # True
    print("o是否是URLRef类型：", isinstance(o, rdflib.term.URIRef)) # False. Jay Karnes为字面量
    print(o.datatype) # 字面量数据类型. 例如 http://www.w3.org/2001/XMLSchema#date