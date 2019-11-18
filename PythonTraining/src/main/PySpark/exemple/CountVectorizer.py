from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer

spark = SparkSession.builder. \
    master("local"). \
    appName("CountVectorizer"). \
    getOrCreate()

df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a".split(" "))
], ["id", "words"])

df2 = spark.createDataFrame([
    (0, "ok good exellent bye".split(" ")),
    (1, "ok bye bye ok exellent ok good ok".split(" "))
], ["id", "words"])

# fit a cv
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=5, minDF=2.0)

model = cv.fit(df2)

result = model.transform(df2)

result.show(truncate=False)
