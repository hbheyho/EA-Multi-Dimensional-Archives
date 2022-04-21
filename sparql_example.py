# 连接D2R进行SPARQL语句查询操作
# 参考链接：https://zhuanlan.zhihu.com/p/32880610

from SPARQLWrapper import SPARQLWrapper, JSON
sparql = SPARQLWrapper("http://localhost:2020/sparql")
sparql.setQuery("""
    PREFIX : <http://www.kgdemo.com#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?n WHERE {
      ?s rdf:type :Person.
      ?s :personName '巩俐'.
      ?s :hasActedIn ?o.
      ?o :movieTitle ?n.
      ?o :movieRating ?r.
    }
    
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for result in results["results"]["bindings"]:
    print(result["n"]["value"])
