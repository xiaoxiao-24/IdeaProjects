package com.oppchain.geospark

import java.awt.Color

import com.vividsolutions.jts.geom.Geometry
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.datasyslab.geospark.formatMapper.shapefileParser.ShapefileReader
import org.datasyslab.geospark.spatialRDD.SpatialRDD
import org.datasyslab.geosparksql.utils.{Adapter, GeoSparkSQLRegistrator}
import org.datasyslab.geosparkviz.core.{ImageGenerator, RasterOverlayOperator}
import org.datasyslab.geosparkviz.core.Serde.GeoSparkVizKryoRegistrator
import org.datasyslab.geosparkviz.extension.visualizationEffect.{ChoroplethMap, HeatMap, ScatterPlot}
import org.datasyslab.geosparkviz.utils.ImageType

object OverlayImageRDD {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("OverlayImage").setMaster("local[*]")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val sparkSession = SparkSession.builder()
      .master("local[*]") // Delete this if run in cluster mode
      .appName("OverlayImage")
      .config("spark.serializer", classOf[KryoSerializer].getName)
      .config("spark.kryo.registrator", classOf[GeoSparkVizKryoRegistrator].getName)
      .getOrCreate()

    GeoSparkSQLRegistrator.registerAll(sparkSession)
    GeoSparkVizRegistrator.registerAll(sparkSession)

    // -----------------------------
    // Taxi file: create DF from CSV
    // -----------------------------

    // Create Spatial DataFrame
    val raw_nytaxi = sparkSession.read.option("header", "true").csv("/Users/xiaoxiaosu/Documents/Codes/sample-data/nyctaxisub.csv")

    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.types._

    val toInt = udf[Int, String](_.toInt)
    val toDouble = udf[Double, String](_.toDouble)

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

    val nyTaxiDf = nyTaxi.select("_id", "_rev", "DropOffTime", "PickUpTime", "DropOffLat", "DropOffLong", "PickUpLat", "PickUpLong", "NbPassenger", "Rate")

    nyTaxiDf.createOrReplaceTempView("tripdf")

    // Convert from DataFrame to RDD. Only Drop off points
    var tripRDD = new SpatialRDD[Geometry]
    tripRDD.rawSpatialRDD = Adapter.toRdd(sparkSession.sql(
      """
        |SELECT ST_PointFromText(concat(DropOffLong, ',', DropOffLat), ',') as shape
        |FROM tripdf
        |where DropOffLong between -74.257159 and -73.699215
        |  and DropOffLat between 40.495992 and 40.915568
        |""".stripMargin))


    // --------------------------------------------
    // NYC Zone file: create RDD->DF from Shapefile
    // --------------------------------------------

    // Shapefile->RDD
    val shapefileInputLocation = "/Users/xiaoxiaosu/Documents/Codes/sample-data/NYC-taxi/MapZone/NYC_TaxiZonesShapeFile"
    //val areaRDD = ShapefileReader.readToGeometryRDD(sparkSession.sparkContext, shapefileInputLocation)
    val areaRDD = ShapefileReader.readToPolygonRDD(sparkSession.sparkContext, shapefileInputLocation)

    // Convert the Coordinate Reference System from degree-based to meter-based. This returns the accurate distance calculate.
    areaRDD.CRSTransform("epsg:4326","epsg:3857")
    tripRDD.CRSTransform("epsg:4326","epsg:3857")

    // !!!NOTE!!!: Analyze RDD to get the rectangle boundary of your dataset and approximate total count.
    areaRDD.analyze()
    tripRDD.analyze()

    // --------------------------------
    // overlay frontimage and backimage
    // --------------------------------
    val imageResolutionX = 1000
    val imageResolutionY = 1000

    val frontImage = new ScatterPlot(imageResolutionX, imageResolutionY, areaRDD.boundaryEnvelope, true)
    frontImage.CustomizeColor(0, 0, 255, 255, Color.GREEN, true) //colorAlpha: opaque or transparent
    frontImage.Visualize(sparkSession.sparkContext, areaRDD)

    val backImage = new HeatMap(imageResolutionX, imageResolutionY, areaRDD.boundaryEnvelope, true, 1)
    //val backImage = new ScatterPlot(imageResolutionX, imageResolutionY, areaRDD.boundaryEnvelope, true)
    backImage.CustomizeColor(255, 0, 0, 50, Color.GREEN, true)  // For heatmap, it doesn't work the color customise???
    backImage.Visualize(sparkSession.sparkContext, tripRDD)

    val overlayOperator = new RasterOverlayOperator(backImage.rasterImage)
    overlayOperator.JoinImage(frontImage.rasterImage)

    val resourceFolder = System.getProperty("user.dir")+"/target/"
    val colocationMapLocation = resourceFolder+"colocationMap_heatmap"
    val imageGenerator = new ImageGenerator
    imageGenerator.SaveRasterImageAsLocalFile(overlayOperator.backRasterImage, colocationMapLocation, ImageType.PNG)

    sparkSession.stop()

  }
}