/*
New York City Taxi trip on 2013 dataset
=======================================

Notes
------
Data Set Characteristics:

    :Number of Instances: 249999

    :Number of Attributes: 20 numeric / 8 categorical

    :Median Value (attribute 14) is usually the target

    :Attribute Information (in order):
        -    DO_Month: drop off month
        -    DO_Week: drop off week of year
        -    DO_WeekDay: drop off day of week
        -    DO_Hour: drop off hour of day (24 hour)
        -    DO_Minute: drop off minute of hour
        -    PU_Month: pick up month
        -    PU_Week: drop off week of year
        -    PU_WeekDay: pick up day of week
        -    PU_Hour: pick up hour of day (24 hour)
        -    PU_Minute: pick up minute of hour
        -    NbPassenger: number of passenger in the trip
        -    hack_license: the driver's license
        -    medallion: the license for taxi
        -    store_and_fwd_flag:
        -    vendor_id: taxi company id
        -    TripDistance: the distance of trip
        -    DropOffLat: drop off latitude
        -    DropOffLong: drop off longitude
        -    PickUpLat: pick up latitude
        -    PickUpLong: pick up longitude
        -    DO_District: drop off district (6 in total)
        -    DO_Zone: drop off zone(260 in total)
        -    DO_Shape_Area
        -    DO_Shape_Leng
        -    PU_District: pick up district (6 in total)
        -    PU_Zone: pick up zone(260 in total)
        -    PU_Shape_Area
        -    PU_Shape_Leng

    :Creator: Xiaoxiao REY, Oppchain
  */

package NycTaxi

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{Normalizer, OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.datasyslab.geospark.formatMapper.shapefileParser.ShapefileReader
import org.datasyslab.geospark.serde.GeoSparkKryoRegistrator
import org.datasyslab.geosparksql.utils.{Adapter, GeoSparkSQLRegistrator}
import org.datasyslab.geosparkviz.core.Serde.GeoSparkVizKryoRegistrator
import org.datasyslab.geosparkviz.sql.utils.GeoSparkVizRegistrator

object Get_train_dataset_NycTaxi {

  def main(args: Array[String]) = {

    val conf = new SparkConf().setAppName("GetData").setMaster("yarn")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val sparkSession = SparkSession.builder()
      .master("yarn") // Delete this if run in cluster mode
      .appName("GetData")
      // Enable GeoSpark custom Kryo serializer
      .config("spark.serializer", classOf[KryoSerializer].getName)
      .config("spark.kryo.registrator", classOf[GeoSparkKryoRegistrator].getName)
      .config("spark.kryo.registrator", classOf[GeoSparkVizKryoRegistrator].getName)
      .getOrCreate()

    // register GeoSpark User Defined Type, User Defined Function and optimized join query strategy
    GeoSparkSQLRegistrator.registerAll(sparkSession)
    GeoSparkVizRegistrator.registerAll(sparkSession)

    // ----------------------------------------
    // Zone file: create RDD->DF from Shapefile
    // ----------------------------------------

    // Shapefile->RDD
    val shapefileInputLocation = "/Users/xiaoxiaosu/Documents/Codes/sample-data/NYC-taxi/MapZone/NYC_TaxiZonesShapeFile"
    val shapefileInputLocation_hdfs = "/user/xiaoxiao/NYC_TaxiZonesShapeFile"
    val spatialRDD_Shapefile = ShapefileReader.readToGeometryRDD(sparkSession.sparkContext, shapefileInputLocation_hdfs)

    // RDD->DF
    val spatialDfShapefile = Adapter.toDf(spatialRDD_Shapefile, sparkSession)


    import org.apache.spark.sql.functions._

    val toInt = udf[Int, Float](_.toInt)
    val toFloat = udf[Float, String](_.toFloat)
    val toDouble = udf[Double, String](_.toDouble)
    val spatialDFraw = spatialDfShapefile.withColumn("ID_Location_raw", toFloat(spatialDfShapefile("location_i")))
      .withColumn("ID_Object_raw", toFloat(spatialDfShapefile("objectid")))
      .withColumn("Shape_Area", toDouble(spatialDfShapefile("shape_area")))
      .withColumn("Shape_Leng", toDouble(spatialDfShapefile("shape_leng")))
    val spatialDFraw_bis = spatialDFraw.withColumn("ID_Location", toInt(spatialDFraw("ID_Location_raw")))
      .withColumn("ID_Object", toInt(spatialDFraw("ID_Location_raw")))
    val spatialDFraw2 = spatialDFraw_bis.select("ID_Location", "ID_Object", "geometry", "borough", "Shape_Area", "Shape_Leng", "zone")

    // Create a Geometry type column and cast the adapt type of columns
    spatialDFraw2.createOrReplaceTempView("rawdf")
    val spatialDf = sparkSession.sql(
      """
        |SELECT borough AS district, zone,
        |       ID_Location, ID_Object, Shape_Area, Shape_Leng,
        |       ST_GeomFromWKT(geometry) AS col_geometry
        |FROM rawdf
            """.stripMargin)
    spatialDf.createOrReplaceTempView("spatialdf")

    // -----------------------------
    // Taxi file: create DF from CSV
    // -----------------------------
    val sourcePath = "/Users/xiaoxiaosu/Documents/Codes/sample-data/nyctaxisub.csv"
    val sourcePathHDFS = "/user/xiaoxiao/nyctaxisub.csv" //"hdfs://35.205.162.209:8020/user/xiaoxiao/nyctaxisub.csv"
    val raw_nytaxi_all = sparkSession.read.option("header", "true").csv(sourcePathHDFS)

    // sample data for local test because it's too big for local
    //val raw_nytaxi = raw_nytaxi_all.sample(0.1, 20000)
    val raw_nytaxi = raw_nytaxi_all

    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.types._

    //val toInt    = udf[Int, String]( _.toInt)
    //val toDouble = udf[Double, String]( _.toDouble)

    val nyTaxi = raw_nytaxi.withColumn("TripDistance",
      toDouble(raw_nytaxi("trip_distance")))
      .withColumn("TripTimeInSecs",
        toInt(raw_nytaxi("trip_time_in_secs")))
      .withColumn("Rate",
        toInt(raw_nytaxi("rate_code")))
      .withColumn("NbPassenger",
        toInt(raw_nytaxi("passenger_count")))
      .withColumn("DropOffLat",
        toDouble(raw_nytaxi("dropoff_latitude")))
      .withColumn("DropOffLong",
        toDouble(raw_nytaxi("dropoff_longitude")))
      .withColumn("PickUpLat",
        toDouble(raw_nytaxi("pickup_latitude")))
      .withColumn("PickUpLong",
        toDouble(raw_nytaxi("pickup_longitude")))
      .withColumn("DropOffTime",
        unix_timestamp(col("dropoff_datetime"), "yyyy-MM-dd HH:mm:ss").cast(TimestampType).as("timestamp"))
      .withColumn("PickUpTime",
        unix_timestamp(col("pickup_datetime"), "yyyy-MM-dd HH:mm:ss").cast(TimestampType).as("timestamp"))

    val nyTaxiDf = nyTaxi.select("_id", "_rev", "DropOffTime", "PickUpTime", "DropOffLat", "DropOffLong", "PickUpLat", "PickUpLong",
      "hack_license", "medallion", "store_and_fwd_flag", "vendor_id", "NbPassenger", "TripDistance", "TripTimeInSecs")
    nyTaxiDf.createOrReplaceTempView("NycTaxiView")


    // -----------------------------------------------
    // transformation of columns
    // -----------------------------------------------

    // 1. Get Polygon Point of the Pick-Up & Drop-Off
    // 2. Transform datetime to plusieur attributes: month, week, day of week etc

    /*val nyTaxiDf_2 = sparkSession.sql(
      """
        |SELECT DropOffTime,
        |       EXTRACT(MONTH FROM DropOffTime) AS DO_Month,
        |       EXTRACT(WEEK FROM DropOffTime) AS DO_Week,
        |       DAYOFWEEK(DropOffTime) AS DO_WeekDay,
        |       EXTRACT(HOUR FROM DropOffTime) AS DO_Hour,
        |       EXTRACT(MINUTE FROM DropOffTime) AS DO_Minute,
        |       PickUpTime,
        |       EXTRACT(MONTH FROM PickUpTime) AS PU_Month,
        |       EXTRACT(WEEK FROM PickUpTime) AS PU_Week,
        |       DAYOFWEEK(PickUpTime) AS PU_WeekDay,
        |       EXTRACT(HOUR FROM PickUpTime) AS PU_Hour,
        |       EXTRACT(MINUTE FROM PickUpTime) AS PU_Minute,
        |       NbPassenger, hack_license, medallion, store_and_fwd_flag, vendor_id,
        |       DropOffLat, DropOffLong, PickUpLat, PickUpLong, TripDistance, TripTimeInSecs,
        |       ST_PointFromText(concat(DropOffLong, ',', DropOffLat), ',') AS polygonDOpoint,
        |       ST_PointFromText(concat(PickUpLong, ',', PickUpLat), ',') AS polygonPUpoint
        |FROM NycTaxiView
      """.stripMargin) */    //sql function only for scala 2.4.4
    val nyTaxiDf_2 = sparkSession.sql(
      """
        |SELECT DropOffTime,
        |       month(DropOffTime) AS DO_Month,
        |       weekofyear(DropOffTime) AS DO_Week,
        |       dayofweek(DropOffTime) AS DO_WeekDay,
        |       hour(DropOffTime) AS DO_Hour,
        |       minute(DropOffTime) AS DO_Minute,
        |       PickUpTime,
        |       month(PickUpTime) AS PU_Month,
        |       weekofyear(PickUpTime) AS PU_Week,
        |       dayofweek(PickUpTime) AS PU_WeekDay,
        |       hour(PickUpTime) AS PU_Hour,
        |       minute(PickUpTime) AS PU_Minute,
        |       NbPassenger, hack_license, medallion, store_and_fwd_flag, vendor_id,
        |       DropOffLat, DropOffLong, PickUpLat, PickUpLong, TripDistance, TripTimeInSecs,
        |       ST_PointFromText(concat(DropOffLong, ',', DropOffLat), ',') AS polygonDOpoint,
        |       ST_PointFromText(concat(PickUpLong, ',', PickUpLat), ',') AS polygonPUpoint
        |FROM NycTaxiView
      """.stripMargin)

    nyTaxiDf_2.createOrReplaceTempView("NycTaxiView_2")

    // --------------------------------------
    // Join Zone with Point
    // --------------------------------------

    // 1. Drop off point
    val NycTaxi_DO_zone = sparkSession.sql(
      """
        |SELECT NycTaxiView_2.*,
        |       spatialdf.district AS DO_District,
        |       spatialdf.zone AS DO_Zone,
        |       spatialdf.ID_Location AS DO_ID_Location,
        |       spatialdf.ID_Object AS DO_ID_Object,
        |       spatialdf.Shape_Area AS DO_Shape_Area,
        |       spatialdf.Shape_Leng AS DO_Shape_Leng,
        |       spatialdf.col_geometry AS DO_Polygon
        |FROM spatialdf, NycTaxiView_2
        |WHERE ST_Contains(spatialdf.col_geometry,NycTaxiView_2.polygonDOpoint)
        |""".stripMargin
    )
    //println("------ the NYC taxi trip join Drop Off zone ------")
    //NycTaxi_DO_zone.show(false)
    NycTaxi_DO_zone.createOrReplaceTempView("NycTaxiView_3")
    //println(NycTaxi_DO_zone.count()) //244193

    // 2. Pick up point
    val NycTaxi_PU_zone = sparkSession.sql(
      """
        |SELECT NycTaxiView_3.*,
        |       spatialdf.district AS PU_District,
        |       spatialdf.zone AS PU_Zone,
        |       spatialdf.ID_Location AS PU_ID_Location,
        |       spatialdf.ID_Object AS PU_ID_Object,
        |       spatialdf.Shape_Area AS PU_Shape_Area,
        |       spatialdf.Shape_Leng AS PU_Shape_Leng,
        |       spatialdf.col_geometry AS PU_Polygon
        |FROM spatialdf, NycTaxiView_3
        |WHERE ST_Contains(spatialdf.col_geometry,NycTaxiView_3.polygonPUpoint)
        |""".stripMargin
    )
    //println("------ the NYC taxi trip join Pick Up zone ------")
    //NycTaxi_PU_zone.printSchema()
    //NycTaxi_PU_zone.show()
    //NycTaxi_PU_zone.createOrReplaceTempView("NycTaxiView_4")
    //println(NycTaxi_PU_zone.count()) //243843

    // ----------------------------------
    // Get train dataset with its columns
    // ----------------------------------
    val train_DF = NycTaxi_PU_zone.select("DO_Month", "DO_Week", "DO_WeekDay", "DO_Hour", "DO_Minute",
      "PU_Month", "PU_Week", "PU_WeekDay", "PU_Hour", "PU_Minute",
      "NbPassenger", "hack_license", "medallion", "store_and_fwd_flag", "vendor_id", "TripDistance",
      "DropOffLat", "DropOffLong", "PickUpLat", "PickUpLong",
      "DO_District", "DO_Zone", "DO_Shape_Area", "DO_Shape_Leng",
      "PU_District", "PU_Zone", "PU_Shape_Area", "PU_Shape_Leng",
      "TripTimeInSecs"
    )

    // ----------------------------------
    // index for category columns
    // ----------------------------------
    // one-hot encoding on our categorical features
    // First we have to use the StringIndexer to convert the strings to integers.
    val indexer1 = new StringIndexer().
      setInputCol("medallion").
      setOutputCol("medallionIndex").
      setHandleInvalid("keep")
    val DF_1 = indexer1.fit(train_DF).transform(train_DF)

    val indexer2 = new StringIndexer().
      setInputCol("hack_license").
      setOutputCol("hack_licenseIndex").
      setHandleInvalid("keep")
    val DF_2 = indexer2.fit(DF_1).transform(DF_1)

    val indexer3 = new StringIndexer().
      setInputCol("store_and_fwd_flag").
      setOutputCol("store_and_fwd_flagIndex").
      setHandleInvalid("keep")
    val DF_3 = indexer3.fit(DF_2).transform(DF_2)

    val indexer4 = new StringIndexer().
      setInputCol("vendor_id").
      setOutputCol("vendor_idIndex").
      setHandleInvalid("keep")
    val DF_4 = indexer4.fit(DF_3).transform(DF_3)

    val indexer5 = new StringIndexer().
      setInputCol("DO_District").
      setOutputCol("DO_DistrictIndex").
      setHandleInvalid("keep")
    val DF_5 = indexer5.fit(DF_4).transform(DF_4)

    val indexer6 = new StringIndexer().
      setInputCol("DO_Zone").
      setOutputCol("DO_ZoneIndex").
      setHandleInvalid("keep")
    val DF_6 = indexer6.fit(DF_5).transform(DF_5)

    val indexer7 = new StringIndexer().
      setInputCol("PU_District").
      setOutputCol("PU_DistrictIndex").
      setHandleInvalid("keep")
    val DF_7 = indexer7.fit(DF_6).transform(DF_6)

    val indexer8 = new StringIndexer().
      setInputCol("PU_Zone").
      setOutputCol("PU_ZoneIndex").
      setHandleInvalid("keep")
    val DF_Indexed = indexer8.fit(DF_7).transform(DF_7)

    // ----------------------------------
    // encode the indexed columns
    // ----------------------------------

    // use the OneHotEncoderEstimator to do the encoding.
    val encoder = new OneHotEncoderEstimator().
      setInputCols(Array("medallionIndex", "hack_licenseIndex", "store_and_fwd_flagIndex", "vendor_idIndex", "DO_DistrictIndex", "DO_ZoneIndex", "PU_DistrictIndex", "PU_ZoneIndex")).
      setOutputCols(Array("medallionVec", "hack_licenseVec", "store_and_fwd_flagVec", "vendor_idVec", "DO_DistrictVec", "DO_ZoneVec", "PU_DistrictVec", "PU_ZoneVec"))
    val DF_encoded = encoder.fit(DF_Indexed).transform(DF_Indexed)

    //DF_encoded.printSchema()
    //DF_encoded.select("medallionVec","hack_licenseVec","store_and_fwd_flagVec",
    //  "vendor_idVec","DO_DistrictVec","DO_ZoneVec","PU_DistrictVec","PU_ZoneVec").show()


    // ----------------------------------
    // set label column
    // ----------------------------------
    // Spark models need exactly two columns: “label” and “features”
    // create label column
    val DF_label = (DF_encoded.select(DF_encoded.col("TripTimeInSecs").as("label"),
      DF_encoded.col("medallionVec"), DF_encoded.col("hack_licenseVec"),
      DF_encoded.col("store_and_fwd_flagVec"), DF_encoded.col("vendor_idVec"),
      DF_encoded.col("DO_DistrictVec"), DF_encoded.col("DO_ZoneVec"),
      DF_encoded.col("PU_DistrictVec"), DF_encoded.col("PU_ZoneVec"),
      DF_encoded.col("DO_Month"), DF_encoded.col("DO_Week"),
      DF_encoded.col("DO_WeekDay"), DF_encoded.col("DO_Hour"), DF_encoded.col("DO_Minute"),
      DF_encoded.col("PU_Month"), DF_encoded.col("PU_Week"),
      DF_encoded.col("PU_WeekDay"), DF_encoded.col("PU_Hour"), DF_encoded.col("PU_Minute"),
      DF_encoded.col("NbPassenger"),
      DF_encoded.col("TripDistance"),
      DF_encoded.col("DO_Shape_Area"), DF_encoded.col("DO_Shape_Leng"),
      DF_encoded.col("PU_Shape_Area"), DF_encoded.col("PU_Shape_Leng"),
      DF_encoded.col("DropOffLat"), DF_encoded.col("DropOffLong"),
      DF_encoded.col("PickUpLat"), DF_encoded.col("PickUpLong")
    ))
    //DF_label.printSchema()
    //DF_label.show(false)

    // ----------------------------------
    // assembler tous les features
    // ----------------------------------

    val assembler = new VectorAssembler().setInputCols(Array("medallionVec", "hack_licenseVec", "store_and_fwd_flagVec",
      "vendor_idVec", "DO_DistrictVec", "DO_ZoneVec", "PU_DistrictVec", "PU_ZoneVec",
      "DO_Month", "DO_Week", "DO_WeekDay", "DO_Hour", "DO_Minute",
      "PU_Month", "PU_Week", "PU_WeekDay", "PU_Hour", "PU_Minute",
      "NbPassenger", "TripDistance",
      "DropOffLat", "DropOffLong", "PickUpLat", "PickUpLong",
      "DO_Shape_Area", "DO_Shape_Leng", "PU_Shape_Area", "PU_Shape_Leng"
    )).
      setOutputCol("features")
    // Transform the DataFrame
    val DF_assemble = assembler.transform(DF_label).select("label", "features")
    //DF_assemble.printSchema()
    //DF_assemble.show()

    // ----------------------------------
    // normaliser les colonnes
    // ----------------------------------
    val normalizer = new Normalizer().setInputCol("features").
      setOutputCol("normFeatures").
      setP(2.0)
    val DF_norm = normalizer.transform(DF_assemble)

    //DF_norm.write.parquet("/Users/xiaoxiaosu/Documents/Codes/ML_training_dataset/nyctaxi_training.parquet")

    // Splitting dataset ( one part for train 70%, one for test 30%)
    val Array(training, test) = DF_norm.randomSplit(Array(0.7, 0.3), seed = 12345)

    import java.time.LocalDateTime
    import java.time.format.DateTimeFormatter
    val file_date = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd"))
    // save data
    val train_path = "/Users/xiaoxiaosu/Documents/Codes/ML_training_dataset/nyctaxi_training.parquet"
    val test_path = "/Users/xiaoxiaosu/Documents/Codes/ML_training_dataset/nyctaxi_test.parquet"
    val train_hdfs_path = "/Dev/ml_train_dataset/nyctaxi_training_"+file_date+".parquet"
    val test_hdfs_path = "/Dev/ml_train_dataset/nyctaxi_test_"+file_date+".parquet"
    training.write.mode(SaveMode.Overwrite).parquet(train_hdfs_path)
    test.write.mode(SaveMode.Overwrite).parquet(test_hdfs_path)

    sparkSession.stop()
    sc.stop()
  }
}
