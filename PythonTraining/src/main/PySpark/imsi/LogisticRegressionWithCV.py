from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderEstimator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from bool2int import bool2int

# Load data
spark = SparkSession.builder. \
    master("local"). \
    appName("Imsi prediction"). \
    getOrCreate()

path = "/Users/xiaoxiaosu/Documents/Codes/sample-data/imsi_json.json"
imsiDF = spark.read.json(path)
imsiDF.createOrReplaceTempView("imsiDF")
spark.udf.register("bool2int", bool2int)
imsi_train = spark.sql("select arpu,averagemonthlybill, totalspent, bool2int(smartphone), "
                       "handsettype, category,type, daysactive, "
                       "dayslastactive, bool2int(canceled)"
                       "from imsiDF "
                       "where canceled is not null and category <> '' "
                       "and type <> '' "
                       "and handsettype <> '' ")

##################################
#  Conversion of string columns  #
##################################

# First we have to use the StringIndexer to convert the string-columns to integer-column.
indexer1 = StringIndexer().setInputCol("handsettype").setOutputCol("handsettypeIndex").setHandleInvalid("keep")
DF_Churn1 = indexer1.fit(imsi_train).transform(imsi_train)

indexer2 = StringIndexer().setInputCol("category").setOutputCol("categoryIndex").setHandleInvalid("keep")
DF_Churn2 = indexer2.fit(DF_Churn1).transform(DF_Churn1)

indexer3 = StringIndexer().setInputCol("type").setOutputCol("typeIndex").setHandleInvalid("keep")
DF_Churn3 = indexer3.fit(DF_Churn2).transform(DF_Churn2)


###########################################
#  encode the Indexed-columns to vectors  #
###########################################
# use the OneHotEncoderEstimator to do the encoding.
encoder = OneHotEncoderEstimator(). \
    setInputCols(["handsettypeIndex", "categoryIndex", "typeIndex"]). \
    setOutputCols(["handsettypeVec", "categoryVec", "typeVec"])
DF_Churn_encoded = encoder.fit(DF_Churn3).transform(DF_Churn3)


########################################
#  Prepare the training-ready dataset  #
########################################
# Spark models need exactly two columns: “label” and “features”
# 1) create "label" column
get_label = DF_Churn_encoded.select(col("bool2int(canceled)").cast("int").alias("label"),
                                    DF_Churn_encoded.arpu, DF_Churn_encoded.averagemonthlybill,
                                    DF_Churn_encoded.totalspent, col("bool2int(smartphone)").cast("int"),
                                    DF_Churn_encoded.daysactive, DF_Churn_encoded.dayslastactive,
                                    DF_Churn_encoded.handsettypeVec, DF_Churn_encoded.categoryVec,
                                    DF_Churn_encoded.typeVec)

# 2) assembler the "features" columns
assembler = VectorAssembler().setInputCols(["arpu", "averagemonthlybill", "totalspent", "bool2int(smartphone)",
                                            "daysactive", "dayslastactive", "handsettypeVec", "categoryVec",
                                            "typeVec"]).setOutputCol("features")

# Transform the DataFrame
output = assembler.transform(get_label).select("label", "features")
(trainingData, testData) = output.randomSplit([0.75, 0.25], 7000)


#################################
#  Train with cross validation  #
#################################
# model tuning by changing the parameters
#lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)
lr = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.01)
#lr = LogisticRegression(maxIter=100, regParam=0.1, elasticNetParam=0.2)

# create the param grid
paramGrid = ParamGridBuilder().build()
# create cross val object, define scoring metric
cv = CrossValidator().setEstimator(lr).setEvaluator(MulticlassClassificationEvaluator().
                                                    setMetricName("weightedRecall")). \
    setEstimatorParamMaps(paramGrid). \
    setNumFolds(5). \
    setParallelism(2)

# fit the model with cross validation
lrModel = cv.fit(trainingData)

########################
#   Make predictions   #
########################
predictions = lrModel.transform(testData)
# Select example rows to display.
predictions.select("features", "label", "prediction", "probability").where("label <> prediction").show(5)

##################
#   evaluation   #
##################
# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

