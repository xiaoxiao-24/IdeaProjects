package NycTaxi

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

object Predictions {

  def main(args: Array[String]) = {

    val conf = new SparkConf().setAppName("MakePrediction").setMaster("yarn")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val sparkSession = SparkSession.builder()
      .master("yarn") // Delete this if run in cluster mode
      .appName("MakePrediction")
      .config("spark.serializer", classOf[KryoSerializer].getName)
      .getOrCreate()

    import java.time.LocalDateTime
    import java.time.format.DateTimeFormatter
    val file_date = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd"))

    // get trained model
    val model_path = "/Users/xiaoxiaosu/Documents/Codes/ML_models/NYCtaxi_RandomForest_2_handleinvalid"
    val model_hdfspath = "/Dev/ml_model/NYCtaxi_RandomForest_"+file_date
    val dtModel = PipelineModel.load(model_hdfspath)

    // get training data (10% sampled)
    val testdata_path = "/Users/xiaoxiaosu/Documents/Codes/ML_training_dataset/nyctaxi_test_20200102104151.parquet"
    val testdata_hdfspath = "/Dev/ml_train_dataset/nyctaxi_test_"+file_date+".parquet"
    val test_DF = sparkSession.read.parquet(testdata_hdfspath)

    // Make predictions.
    val predictions = dtModel.transform(test_DF)

    // write predictions to (parquet) HDFS
    val predictions_hdfspath = "/Dev/ml_predictions/Prediction_DecisionTree"+file_date+".parquet"
    predictions.select("prediction", "label", "indexedFeatures")
               .write
               .mode(SaveMode.Overwrite)
               .format("parquet")
               .save(predictions_hdfspath)

    /*
    // Select example rows to display.
    predictions.select("prediction", "label", "indexedFeatures").show(20)
    predictions.createOrReplaceTempView("predictions")
    sparkSession.sql("select count(distinct label) from predictions ").show()
    sparkSession.sql("select label,count(1) from predictions group by label order by count(1) desc").show()
    */

    // Select (prediction, true label) and compute test error: Root Mean Squared Error (RMSE).
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    //println(s"Root Mean Squared Error (RMSE) on test data = $rmse")  // Decision Tree: 475.9471977969154

    // write Root Mean Squared Error to (parquet) HDFS
    import sparkSession.implicits._

    //case class RMSE(Root_Mean_Squared_Error: Double)
    val rmseDF = Seq((rmse)).toDF("Root_Mean_Squared_Error")
    val rmse_hdfspath = "/Dev/ml_predictions/RMSE_DecisionTree"+file_date+".parquet"
    rmseDF.write.mode(SaveMode.Overwrite).format("parquet").save(rmse_hdfspath)

    sparkSession.stop()
    sc.stop()

  }

}
