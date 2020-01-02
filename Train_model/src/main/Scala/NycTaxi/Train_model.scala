package NycTaxi

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object Train_model {
  def main(args: Array[String]) = {

    val conf = new SparkConf().setAppName("TrainModel").setMaster("yarn")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val sparkSession = SparkSession.builder()
      .master("yarn") // Delete this if run in cluster mode
      .appName("TrainModel")
      .getOrCreate()

    import java.time.LocalDateTime
    import java.time.format.DateTimeFormatter
    val file_date = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd"))

    // get training data (10% sampled)
    val train_data_path = "/Users/xiaoxiaosu/Documents/Codes/ML_training_dataset/nyctaxi_training.parquet"
    val train_data_hdfspath = "/Dev/ml_train_dataset/nyctaxi_training_"+file_date+".parquet"
    val train_DF = sparkSession.read.parquet(train_data_hdfspath)

    // Automatically identify categorical features, and index them.
    // Here, we treat features with > 4 distinct values as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("normFeatures")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .setHandleInvalid("skip") //keep, skip or error
      .fit(train_DF)

    // Decision Tree
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dt))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(train_DF)

    val model_path = "/Users/xiaoxiaosu/Documents/Codes/ML_models/NYCtaxi_RandomForest_"+file_date
    val model_hdfspath = "/Dev/ml_model/NYCtaxi_RandomForest_"+file_date
    model.write.overwrite().save(model_hdfspath)

    sparkSession.stop()
    sc.stop()

  }
}
