##############################################
#      Multi-Class Text Classification       #
#    Logistic Regression with CountVector    #
##############################################

# Input: Descript
# Output: Category

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, StringIndexer, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

sc = SparkContext()
sc.setLogLevel("ERROR")

spark = SparkSession.builder. \
    master("local"). \
    appName("san francisco crime prediction"). \
    getOrCreate()

##################################
#       read from CSV file       #
##################################
path = "/Users/xiaoxiaosu/Documents/Codes/sample-data/sf-crime/train.csv"
data_all = spark.read.options(header='true', inferschema='true').csv(path)

##################################
#      Description of CSV        #
##################################
print("Train data set schema: ")
data_all.printSchema()
print("Number of test data set: " + str(data_all.count()))
data_all.show(5)

###################################
#      Selection of columns       #
###################################
drop_list = ['Dates', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']

data_raw = data_all.select([column for column in data_all.columns if column not in drop_list])
data_raw.printSchema()
print("Columns in train dataset are " + str(data_raw.columns))
data_raw.show(5)

#######################################
#      Description of variables       #
#######################################
print(" \n--- Top 20 Categories ---\n ")
data_raw.groupBy("Category").count().orderBy(col("count").desc()).show(truncate=False)
print(" \n --- Top 20 Descript ---\n")
data_raw.groupBy("Descript").count().orderBy(col("count").desc()).show(truncate=False)

#################################
#      feature extraction       #
#################################
# feature extraction in 2 parties

# column : features
# 1.regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="Descript", outputCol="words", pattern="\\W")
# 2.stop words
add_stopwords = ["http", "https", "amp", "rt", "t", "c", "the"]
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")\
    .setStopWords(add_stopwords)
# 3.bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
# 3.TF-IDF features
#hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
#idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)

# column : label
# index the label column
label_stringIdx = StringIndexer(inputCol="Category", outputCol="label")

###################################
#      Pipeline for data set      #
###################################
# create pipeline for data preparation
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
#pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx])

# fit the pipeline
pipelineFit = pipeline.fit(data_raw)
data = pipelineFit.transform(data_raw)
print("\nDataset ready for train: ")
data.show(truncate=False)

#######################################
#     split Train & Test dataset      #
#######################################
(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=1000)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

#####################
#     training      #
#####################
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
#lrModel = lr.fit(trainingData)

#############################
#     cross-validation      #
#############################
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.3, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) \
    .addGrid(lr.maxIter, [10, 20, 50]) \
    .build()

evaluator = MulticlassClassificationEvaluator()

cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)

cvModel = cv.fit(trainingData)

#######################
#     prediction      #
#######################
#predictions = lrModel.transform(testData)
predictions = cvModel.transform(testData)

#    .filter(predictions['prediction'] == 0)\
predictions\
    .select("Descript", "Category", "probability", "label", "prediction")\
    .orderBy("probability", ascending=False)\
    .show(n=50, truncate=50)

#######################
#     evaluation      #
#######################
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
# Test Error(Logistic regression + CountVector) = 0.0270472
# Test Error(Logistic regression +  TF-IDF) = 0.0271614
# Test Error(Logistic regression + Cross Validation) = 0.00774494
# Test Error(Naive Bayes + CountVector) = 0.00641731

sc.stop()
spark.stop()




