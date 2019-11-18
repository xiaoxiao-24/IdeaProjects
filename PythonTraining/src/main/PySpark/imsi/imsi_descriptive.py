##########################
#  Descriptive analysis  #
##########################

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
from datetime import date

conf = SparkConf().setAppName("Sparkimsi").setMaster("local[*]")
sc = SparkContext()

spark = SparkSession.builder. \
    master("local"). \
    appName("Imsi prediction"). \
    getOrCreate()

sc.setLogLevel("ERROR")

today = date.today()

##################################
#      read from JSON file       #
##################################
path = "/Users/xiaoxiaosu/Documents/Codes/sample-data/imsi_json.json"
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


##########################################################
#      distribution and relationships of variables       #
##########################################################
# distplot(): Plot a distribution plot using seaborn's distplot() method.
sample_df = imsi_train.select(['daysactive']).sample(False, 0.2, 42)
pandas_df = sample_df.toPandas()
sns.distplot(pandas_df)
path_fig = "/Users/xiaoxiaosu/Downloads/"
nom_fig = "imsi_daysactive_" + str(today) + ".png"
save_to = path_fig + nom_fig
plt.savefig(save_to, bbox_inches='tight')
plt.show()

# lmplot(): a linear model plots helps visualize
#           if variables have relationships with the dependent variable.
sample2_df = imsi_train.select(['arpu', 'daysactive']).sample(False, 0.2, 42)
pandas2_df = sample2_df.toPandas()
sns.lmplot(x='arpu', y='daysactive', data=pandas2_df)
path_fig = "/Users/xiaoxiaosu/Downloads/"
nom_fig = "imsi_daysactive_vs_arpu_" + str(today) + ".png"
save_to = path_fig + nom_fig
plt.savefig(save_to, bbox_inches='tight')
plt.show()
