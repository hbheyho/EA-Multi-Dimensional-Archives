# 连接jena进行SPARQL语句查询和推理操作
# 参考链接：https://zhuanlan.zhihu.com/p/33224431

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://localhost:3030/kg_demo_movie/sparql")
# 会根据推理规则找出所有喜剧演员
# rules: [ruleComedian: (?p :hasActedIn ?m) (?m :hasGenre ?g) (?g :genreName '喜剧') -> (?p rdf:type :Comedian)]

# PREFIX : <http://www.kgdemo.com#>
# PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
# PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
#
# SELECT * WHERE {
# ?x :movieTitle '功夫'.
# ?x ?p ?o.
# }

sparql.setQuery(r'''
PREFIX : <http://www.kgdemo.com#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT * WHERE {
    ?x rdf:type :Comedian.
    ?x :personName ?n.
    }
    limit 20
    
    ''')

sparql.setReturnFormat(JSON)
results = sparql.query().convert()

print(results,"\n")

for result in results["results"]["bindings"]:
    print(result)
