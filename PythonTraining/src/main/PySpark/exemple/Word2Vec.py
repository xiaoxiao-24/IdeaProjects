from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec

spark = SparkSession.builder. \
    master("local"). \
    appName("Word2Vec"). \
    getOrCreate()

documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "), ),
    ("I wish Java could use case classes".split(" "), ),
    ("Logistic regression models are neat".split(" "), )
], ["text"])

# Learn a mapping from words to Vectors
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)
print(word2Vec)
print(model)

result = model.transform(documentDF)
print(result)
result.show()

for row in result.collect():
    text, vector = row
    print(row)
    print(text)
    print(vector)
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
