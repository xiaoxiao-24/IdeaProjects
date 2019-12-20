##############################################
#      Multi-Class Text Classification       #
#      text mining / sentiment prediction      #
##############################################

import datetime as dt
from pyspark import SparkContext
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import functions
from pyspark.ml.classification import NaiveBayes, LogisticRegression, RandomForestClassifier, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext()
sc.setLogLevel("ERROR")

spark = SparkSession.builder. \
    master("local"). \
    appName("Women clothing commerce review"). \
    getOrCreate()

##################################
#       read from CSV file       #
##################################
path = "/Users/xiaoxiaosu/Documents/Codes/sample-data/WomensClothingE-CommerceReviews/WomensClothingE-CommerceReviews" \
       ".csv"
data_all = spark.read.options(header='true', inferschema='true', delimiter=',').csv(path)

print("\nNumber of lines: " + str(data_all.count()))

data_rename = data_all.select(col("Clothing ID").alias("ClothingID"),
                              col("Age"),
                              col("Title"),
                              col("Review Text").alias("ReviewText"),
                              col("Rating"),
                              col("Recommended IND").alias("RecommendedIND"),
                              col("Positive Feedback Count").alias("PositiveFeedbackCount"),
                              col("Division Name").alias("DivisionName"),
                              col("Department Name").alias("DepartmentName"),
                              col("Class Name").alias("ClassName")
                              )

data_rename.show()

# fill NA values by ''
data_rename.createOrReplaceTempView("data_rename")
data_no_na = spark.sql("select * from data_rename where ucase(ReviewText) not like '%NA%' ")

print(" ---- Remove NA ---- ")
data_no_na.show(10, truncate=False)

# 1.regular expression tokenizer
tokenizer = Tokenizer(inputCol="ReviewText", outputCol="words")
data_token = tokenizer.transform(data_no_na)
print(" ---- tokenizer ---- ")
data_token.show(10, truncate=False)
# 2.stop words
add_stopwords = ["http", "https", "are", "was", "is", "the", "i", "this", "that"]
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered") \
    .setStopWords(add_stopwords)
data_filtered = stopwordsRemover.transform(data_token)
print(" ---- filter stop word ---- ")
data_filtered.show(10, truncate=False)
# 3.bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
data_features = countVectors.fit(data_filtered).transform(data_filtered)
print(" ---- features (vector) ---- ")
data_features.show(10, truncate=False)
# 4.label
# data_label = data_features.select(functions.when(col("Rating")>=4, 1).when(col("Rating")<=4,-1).otherwise(0))
data_label_raw = data_features.filter(col("Rating") != 3)
data_label = data_label_raw.withColumn("label",
                                       functions.when(col("Rating") >= 4, 1).otherwise(
                                           0))  # when(col("Rating") <= 3, -1)

(train, test) = data_label.randomSplit([0.7, 0.3], seed=1000)
"""
##########################
#     training bayes     #
##########################
#Test Error = 0.0844938
start = dt.datetime.now()
nb = NaiveBayes(smoothing=2)
Model = nb.fit(train)
print("Elapsed time Bayes: ", str(dt.datetime.now() - start))
"""
########################################
#     training logistic regression     #
########################################
#Test Error = 0.102431
start = dt.datetime.now()
lr = LogisticRegression()
Model = lr.fit(train)
print("Elapsed time Logistic Regression: ", str(dt.datetime.now()-start))
"""
##################################
#     training Random Forest     #
##################################
#Test Error = 0.191935
start = dt.datetime.now()
rf = RandomForestClassifier()
Model = rf.fit(train)
print("Elapsed time Random Forest: ", str(dt.datetime.now()-start))
###########################################
#     training one-vs-rest classifier     #
###########################################
start = dt.datetime.now()
lr = LogisticRegression()
ovr = OneVsRest(classifier=lr)
Model = ovr.fit(train)
print("Elapsed time Random Forest: ", str(dt.datetime.now()-start))
"""
#######################
#     prediction      #
#######################
predictions = Model.transform(test)

predictions.printSchema()
print("TP and TN: " + str(predictions.where(col("label") == col("prediction")).count()))
print("FP and FN: " + str(predictions.where(col("label") != col("prediction")).count()))
predictions.select("label", "prediction") \
    .orderBy("probability", ascending=False).show(n=10, truncate=30)

#######################
#     evaluation      #
#######################
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("\nTest Error = %g" % (1.0 - accuracy))


sc.stop()

spark.stop()
