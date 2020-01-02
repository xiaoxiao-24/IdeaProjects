package com.oppchain.geospark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.datasyslab.geospark.serde.GeoSparkKryoRegistrator
import org.datasyslab.geosparkviz.core.Serde.GeoSparkVizKryoRegistrator

object NycTaxiPrediction {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("BestRoute").setMaster("local[*]")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val sparkSession = SparkSession.builder()
      .master("local[*]") // Delete this if run in cluster mode
      .appName("BestRoute")
      // Enable GeoSpark custom Kryo serializer
      .config("spark.serializer", classOf[KryoSerializer].getName)
      .config("spark.kryo.registrator", classOf[GeoSparkKryoRegistrator].getName)
      .config("spark.kryo.registrator", classOf[GeoSparkVizKryoRegistrator].getName)
      .getOrCreate()

    // get trained model
    val dtModel = PipelineModel.load("/Users/xiaoxiaosu/Documents/Codes/ML_models/NYCtaxi_RandomForest")

    // get training data (10% sampled)
    val test_DF = sparkSession.read.parquet("/Users/xiaoxiaosu/Documents/Codes/ML_training_dataset/nyctaxi_test.parquet")

    // Make predictions.
    val predictions = dtModel.transform(test_DF)

    // Select example rows to display.
    predictions.select("prediction", "label", "indexedFeatures").show(20)
    predictions.createOrReplaceTempView("predictions")
    sparkSession.sql("select count(distinct label) from predictions ").show()
    sparkSession.sql("select label,count(1) from predictions group by label order by count(1) desc").show()

    // Select (prediction, true label) and compute test error: Root Mean Squared Error (RMSE).
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")  // Decision Tree: 475.9471977969154

    sparkSession.stop()
    sc.stop()

  }
}
