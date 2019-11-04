#####################
#   Random Forest   #
#####################

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoderEstimator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix

conf = SparkConf().setAppName("Sparkimsi").setMaster("local[*]")
sc = SparkContext()

spark = SparkSession.builder. \
    master("local"). \
    appName("Imsi prediction"). \
    getOrCreate()

sc.setLogLevel("ERROR")

##################################
#      read from JSON file       #
##################################
path = "/Users/xiaoxiaorey/IdeaProjects/sparktraining/data_source/imsi_json.json"
imsiDF = spark.read.json(path)

imsiDF.createOrReplaceTempView("imsiDF")


def bool2int(var):
    if var:
        return 1
    else:
        return 0


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

# methodes 2: change column name
# get_label = DF_Churn_encoded.withColumnRenamed("bool2int(canceled)", "label")

# 2) assembler the "features" columns
assembler = VectorAssembler().setInputCols(["arpu", "averagemonthlybill", "totalspent", "bool2int(smartphone)",
                                            "daysactive", "dayslastactive", "handsettypeVec", "categoryVec",
                                            "typeVec"]).setOutputCol("features")

# Transform the DataFrame
output = assembler.transform(get_label).select("label", "features")


############################################################
#  seperate train dataset into 2 parts: training and test  #
############################################################
# prepare dataset ( one part for train 70%, one for test 30%)
# Split data into training (70%) and test (30%)
[training, test] = output.select("label", "features").randomSplit([0.7, 0.3], 12345)
training.cache()


#####################
#  train the model  #
#####################
# create the training model
rf = RandomForestClassifier()

# create the param grid
paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [20, 50, 100]).build()

# create cross val object, define scoring metric
cv = CrossValidator().setEstimator(rf).setEvaluator(MulticlassClassificationEvaluator().
                                                    setMetricName("weightedRecall")). \
    setEstimatorParamMaps(paramGrid). \
    setNumFolds(3). \
    setParallelism(2)

# train the model
# You can then treat this object as the model and use fit on it.
model = cv.fit(training)


##############################################
#  interpretation of the result of training  #
##############################################

# get the results of training, test with the test dataset
results = model.transform(test).select("features", "label", "prediction")

print("model results schema: ")
results.printSchema()

print("model results: ")
results.show()

# convert these results(get "prediction" and "label" to compare) to an RDD
predictionAndLabels = results.select(col("prediction").cast("double"), col("label").cast("double")).rdd

print("prediction v.s. labels: ")
predictionAndLabels.take(5)

# create our metrics objects and print out the confusion matrix.
# Instantiate a new metrics objects
bMetrics = BinaryClassificationMetrics(predictionAndLabels)
mMetrics = MulticlassMetrics(predictionAndLabels)

# Print out the Confusion matrix
print("Confusion matrix:")
print(mMetrics.confusionMatrix)

# Overall statistics
precision = mMetrics.precision()
recall = mMetrics.recall()
f1Score = mMetrics.fMeasure()
print("Summary Stats")
print("Precision = %s" % precision)
print("Recall = %s" % recall)
print("F1 Score = %s" % f1Score)

# Statistics by class
labels = predictionAndLabels.map(lambda lp: lp.label).distinct().collect()
# labels = results.select("label").distinct()
for label in sorted(labels):
    print("Class %s precision = %s" % (label, mMetrics.precision(label)))
    print("Class %s recall = %s" % (label, mMetrics.recall(label)))
    print("Class %s F1 Measure = %s" % (label, mMetrics.fMeasure(label, beta=1.0)))

# Weighted stats
print("Weighted recall = %s" % mMetrics.weightedRecall)
print("Weighted precision = %s" % mMetrics.weightedPrecision)
print("Weighted F(1) Score = %s" % mMetrics.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % mMetrics.weightedFMeasure(beta=0.5))
print("Weighted false positive rate = %s" % mMetrics.weightedFalsePositiveRate)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                              predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(results)
print("Test Error = %g" % (1.0 - accuracy))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, verticalalignment="bottom")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 #verticalalignment="center",
                 color="black" if cm[i, j] > thresh else "black",
                 bbox=dict(facecolor='white', alpha=1))

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


y_true = results.select("label")
y_true = y_true.toPandas()

y_pred = results.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
print(cnf_matrix)

# Plot non-normalized confusion matrix
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(121)
ax.set_aspect('equal')
plot_confusion_matrix(cnf_matrix, classes=labels,
                      title='Confusion matrix, without normalization')
# plt.show()

# Plot normalized confusion matrix
# plt.figure()
ax = fig.add_subplot(122)
ax.set_aspect('equal')
plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
