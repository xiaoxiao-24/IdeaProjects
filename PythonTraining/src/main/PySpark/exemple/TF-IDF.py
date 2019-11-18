from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

spark = SparkSession.builder. \
    master("local"). \
    appName("TF-IDF"). \
    getOrCreate()

sentenceData = spark.createDataFrame([
    (0.0, "Hi I heard about Spark"),
    (0.0, "I wish Java could use case classes"),
    (1.0, "Logistic regression models are neat")
], ["label", "sentence"])
print(sentenceData)
sentenceData.show()

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)
print(wordsData)
wordsData.show()

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)
print(featurizedData)
featurizedData.show()

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
print(rescaledData)
rescaledData.show()

rescaledData.select("label", "features").show()
