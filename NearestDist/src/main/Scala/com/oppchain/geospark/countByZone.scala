package com.oppchain.geospark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.datasyslab.geospark.serde.GeoSparkKryoRegistrator
import org.datasyslab.geosparksql.utils.{Adapter, GeoSparkSQLRegistrator}
import org.datasyslab.geospark.formatMapper.shapefileParser.ShapefileReader
import org.datasyslab.geosparkviz.core.Serde.GeoSparkVizKryoRegistrator

object countByZone {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("NYCTaxiCountByZone").setMaster("local[*]")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val sparkSession = SparkSession.builder()
      .master("local[*]") // Delete this if run in cluster mode
      .appName("NYCTaxiCountByZone")
      // Enable GeoSpark custom Kryo serializer
      .config("spark.serializer", classOf[KryoSerializer].getName)
      .config("spark.kryo.registrator", classOf[GeoSparkKryoRegistrator].getName)
      .config("spark.kryo.registrator", classOf[GeoSparkVizKryoRegistrator].getName)
      .getOrCreate()

    // register GeoSpark User Defined Type, User Defined Function and optimized join query strategy
    GeoSparkSQLRegistrator.registerAll(sparkSession)


    // ----------------------------------------
    // Zone file: create RDD->DF from Shapefile
    // ----------------------------------------

    // Shapefile->RDD
    val shapefileInputLocation = "/Users/xiaoxiaosu/Documents/Codes/sample-data/NYC-taxi/MapZone/NYC_TaxiZonesShapeFile"
    val spatialRDD_Shapefile = ShapefileReader.readToGeometryRDD(sparkSession.sparkContext, shapefileInputLocation)

    // RDD->DF
    val spatialDfShapefile = Adapter.toDf(spatialRDD_Shapefile, sparkSession)
    println("--- Schema of spatialDf from Shapefile ---")
    //spatialDfShapefile.printSchema()
    println("--- Top 5 lines of spatialDfShapefile ---")
    //spatialDfShapefile.show(5)

    // Create a Geometry type column
    spatialDfShapefile.createOrReplaceTempView("rawdf")
    val spatialDf = sparkSession.sql(
      """
        |SELECT ST_GeomFromWKT(geometry) AS col_geometry, borough,
        |       location_i, objectid, shape_area, shape_leng, zone
        |FROM rawdf
        """.stripMargin)
    println("--- Schema of spatialDF ---")
    //spatialDf.printSchema()
    spatialDf.createOrReplaceTempView("spatialdf")
    println("--- Top 5 lines of spatialDf in degree ---")
    //spatialDf.show(5)

    // -----------------------------
    // Taxi file: create DF from CSV
    // -----------------------------
    val raw_nytaxi = sparkSession.read.option("header","true").csv("/Users/xiaoxiaosu/Documents/Codes/sample-data/nyctaxisub.csv")
    println("--- the NYC taxi file schema ---")
    //raw_nytaxi.printSchema()

    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.types._

    val toInt    = udf[Int, String]( _.toInt)
    val toDouble = udf[Double, String]( _.toDouble)
    val toString = udf[String, String](_.toString)

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
                       //.withColumn("DropOffLoc",
                       //  concat(toString(raw_nytaxi("dropoff_latitude")),toString(raw_nytaxi("dropoff_longitude"))))
                       .withColumn("PickUpLat",
                                    toDouble(raw_nytaxi("pickup_latitude")))
                       .withColumn("PickUpLong",
                                    toDouble(raw_nytaxi("pickup_longitude")))
                       //.withColumn("PickUpLoc",
                        //            concat(toString(toDouble(raw_nytaxi("pickup_latitude"))),toString(toDouble(raw_nytaxi("pickup_longitude")))))
                       .withColumn("DropOffTime",
                                    unix_timestamp(col("dropoff_datetime"), "yyyy-MM-dd HH:mm:ss").cast(TimestampType).as("timestamp"))
                       .withColumn("PickUpTime",
                                    unix_timestamp(col("pickup_datetime"), "yyyy-MM-dd HH:mm:ss").cast(TimestampType).as("timestamp"))

    val nyTaxiDf = nyTaxi.select("_id", "_rev", "DropOffTime", "PickUpTime", "DropOffLat", "DropOffLong", "PickUpLat", "PickUpLong", "NbPassenger", "Rate")
    println("--- the NYC taxi file preprocessed schema ---")
    //nyTaxiDf.printSchema()
    //nyTaxiDf.show()

    nyTaxiDf.createOrReplaceTempView("NycTaxiView")

    // -----------------------------------------------
    // Polygon Point of the Pick-Up & Drop-Off
    // -----------------------------------------------
    val NycTaxi_DO_PU_point = sparkSession.sql(
      """
        |SELECT DropOffTime, PickUpTime, NbPassenger, Rate,
        |       ST_PointFromText(concat(DropOffLong, ',', DropOffLat), ',') AS polygondropoffpoint,
        |       ST_PointFromText(concat(PickUpLong, ',', PickUpLat), ',') AS polygonpickuppoint
        |FROM NycTaxiView
      """.stripMargin)
    println("--- the NYC taxi Pick-Up and Drop-Off polygon-point ---")
    NycTaxi_DO_PU_point.show(false)

    NycTaxi_DO_PU_point.createOrReplaceTempView("DO_PU_Point")

    // --------------------------------------
    // Zone of the Drop-Off and Pick-up point
    // --------------------------------------
    val NycTaxi_DO_zone = sparkSession.sql(
      """
        |SELECT DO_PU_Point.polygondropoffpoint, DO_PU_Point.NbPassenger, DO_PU_Point.Rate,
        |       spatialdf.borough, spatialdf.zone, spatialdf.col_geometry
        |FROM spatialdf, DO_PU_Point
        |WHERE ST_Contains(spatialdf.col_geometry,DO_PU_Point.polygondropoffpoint)
        |""".stripMargin
    )
    println("------ the NYC taxi Drop off intersections ------")
    NycTaxi_DO_zone.show(false)
    NycTaxi_DO_zone.createOrReplaceTempView("DropOff_PointZone")

    val NycTaxi_PU_zone = sparkSession.sql(
      """
        |SELECT DO_PU_Point.polygonpickuppoint, DO_PU_Point.NbPassenger, DO_PU_Point.Rate,
        |       spatialdf.borough, spatialdf.zone, spatialdf.col_geometry
        |FROM spatialdf, DO_PU_Point
        |WHERE ST_Contains(spatialdf.col_geometry,DO_PU_Point.polygonpickuppoint)
        |""".stripMargin
    )
    println("------ the NYC taxi Pick up intersections ------")
    NycTaxi_PU_zone.show(false)
    NycTaxi_PU_zone.createOrReplaceTempView("PickUp_PointZone")

    // -----------------------------------
    // TOP Drop Off / Pick Up point of NYC
    // -----------------------------------
    val NycTaxi_DO_by_zone = sparkSession.sql(
      """
        |SELECT borough,count(*),sum(NbPassenger),avg(Rate)
        |FROM DropOff_PointZone
        |GROUP BY borough
        |ORDER BY count(*) DESC
        |""".stripMargin
    )
    println("--- NYC taxi: the TOP Drop Off Zone ---")
    NycTaxi_DO_by_zone.show(false)

    val NycTaxi_PU_by_zone = sparkSession.sql(
      """
        |SELECT borough,count(*),sum(NbPassenger),avg(Rate)
        |FROM PickUp_PointZone
        |GROUP BY borough
        |ORDER BY count(*) DESC
        |""".stripMargin
    )
    println("--- NYC taxi: the TOP Pick Up Zone ---")
    NycTaxi_PU_by_zone.show(false)

    // -----------------------------------------
    // TOP Drop Off / Pick Up point of Manhattan
    // -----------------------------------------
    val NycTaxi_DO_Man = sparkSession.sql(
      """
        |SELECT zone,count(*),sum(NbPassenger),avg(Rate)
        |FROM DropOff_PointZone
        |WHERE borough = 'Manhattan'
        |GROUP BY zone
        |ORDER BY count(*) DESC
        |""".stripMargin
    )
    println("--- NYC taxi: the TOP Drop Off Zone in Manhattan ---")
    NycTaxi_DO_Man.show(false)

    val NycTaxi_PU_Man = sparkSession.sql(
      """
        |SELECT zone,count(*),sum(NbPassenger),avg(Rate)
        |FROM PickUp_PointZone
        |WHERE borough = 'Manhattan'
        |GROUP BY zone
        |ORDER BY count(*) DESC
        |""".stripMargin
    )
    println("--- NYC taxi: the TOP Pick Up Zone in Manhattan ---")
    NycTaxi_PU_Man.show(false)

    sparkSession.stop()
  }
}
