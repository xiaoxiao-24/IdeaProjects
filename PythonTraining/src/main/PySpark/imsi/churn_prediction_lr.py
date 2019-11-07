#########################
#  Logistic Regression  #
#########################

import matplotlib.pyplot as plt
import pylab as p
import pandas as pd
import numpy as np
import seaborn as sns
from bool2int import bool2int
from dataset_prepare import get_data_ready
from show_confusion_matrix import plot_confusion_matrix
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorIndexer, StringIndexer, IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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

"""
def bool2int(var):
    if var:
        return 1
    else:
        return 0
"""

spark.udf.register("bool2int", bool2int)

imsi_train = spark.sql("select arpu, averagemonthlybill, totalspent, bool2int(smartphone), "
                       "handsettype, category,type, daysactive, "
                       "dayslastactive, bool2int(canceled)"
                       "from imsiDF "
                       "where canceled is not null and category <> '' "
                       "and type <> '' "
                       "and handsettype <> '' ")

imsi_train.show(5)

"""
##########################################################
#      distribution and relationships of variables       #
##########################################################
# distplot(): Plot a distribution plot using seaborn's distplot() method.
sample_df = imsi_train.select(['daysactive']).sample(False, 0.2, 42)
pandas_df = sample_df.toPandas()
sns.distplot(pandas_df)
plt.show()

# lmplot(): a linear model plots helps visualize
#           if variables have relationships with the dependent variable.
sample2_df = imsi_train.select(['arpu', 'daysactive']).sample(False, 0.2, 42)
pandas2_df = sample2_df.toPandas()
sns.lmplot(x='arpu', y='daysactive', data=pandas2_df)
plt.show()
"""

#######################################
#      normalisation of dataset       #
#######################################
cat_cols = ['handsettype', 'category', 'type']
num_cols = ['arpu', 'averagemonthlybill', 'totalspent',
            'daysactive', 'dayslastactive']
label_Col = 'bool2int(canceled)'

data = get_data_ready(imsi_train, cat_cols, num_cols, label_Col)
data.show(5)

###########################################
#  encode the feature-columns to vectors  #
###########################################
# Index labels, adding metadata to the label column
labelIndexer = StringIndexer(inputCol='label',
                             outputCol='indexedLabel').fit(data)
labelIndexer.transform(data).show(5, True)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features",
                               outputCol="indexedFeatures",
                               maxCategories=4).fit(data)
featureIndexer.transform(data).show(5, True)

############################################
#  split dataset into : training and test  #
############################################
(trainingData, testData) = data.randomSplit([0.6, 0.4], 10000)

trainingData.show(5, False)
testData.show(5, False)

############################
#  fit the training model  #
############################
logr = LogisticRegression(featuresCol='indexedFeatures', labelCol='indexedLabel')

############################
#  Pipeline Architecture   #
############################
# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, logr, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

########################
#   Make predictions   #
########################
predictions = model.transform(testData)
# Select example rows to display.
predictions.select("features", "label", "predictedLabel").show(5)

##################
#   evaluation   #
##################
# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

lrModel = model.stages[2]
trainingSummary = lrModel.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# Obtain the receiver-operating characteristic as a DataFrame and areaUnderROC.
trainingSummary.roc.show(5)
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))
# plot ROC curve
fpr = trainingSummary.roc.select("FPR").toPandas().values.tolist()
tpr = trainingSummary.roc.select("TPR").toPandas().values.tolist()
plt.title('ROC Curve', verticalalignment="bottom")
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.plot(fpr, tpr)
# fill the area under ROC
# method 1: fill_between()
# fpr_np = np.asarray(fpr)
# tpr_np = np.asarray(tpr)
# plt.fill_between(fpr_np, tpr_np, color='gray') # fill_between() ne marche pas
# method 2: fill()
p.fill(fpr, tpr, color='gray', alpha=0.2)  # fill only une partie
p.show()
plt.show()

# Set the model threshold to maximize F-Measure
fMeasure = trainingSummary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head(5)
# bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
#     .select('threshold').head()['threshold']

# lr.setThreshold(bestThreshold)


#####################
#   visualization   #
#####################

y_true = predictions.select("label")
y_true = y_true.toPandas()

y_pred = predictions.select("predictedLabel")
y_pred = y_pred.toPandas()

class_temp = predictions.select("label").groupBy("label") \
    .count().sort('count', ascending=False).toPandas()
labels = class_temp["label"].values.tolist()

cnf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
print(cnf_matrix)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels,
                      title='Confusion matrix, without normalization')
plt.show()

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

sc.stop()
spark.stop()
