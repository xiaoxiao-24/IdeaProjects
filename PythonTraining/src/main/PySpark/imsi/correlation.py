#################
#  Correlation  #
#################

from pyspark.sql import SparkSession
from pyspark.ml.stat import Correlation
from bool2int import bool2int
from dataset_prepare import get_data_ready
import seaborn as sns
import matplotlib.pyplot as plt


spark = SparkSession.builder. \
    master("local"). \
    appName("Imsi prediction"). \
    getOrCreate()

##################################
#      read from JSON file       #
##################################
path = "/Users/xiaoxiaorey/IdeaProjects/sparktraining/data_source/imsi_json.json"
imsiDF = spark.read.json(path)
imsiDF.show()

#imsiPD = imsiDF.toPandas()
#imsiPD.head(5)

"""
r = imsiPD.corr()

ax = sns.heatmap(
    imsiDF,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()
"""