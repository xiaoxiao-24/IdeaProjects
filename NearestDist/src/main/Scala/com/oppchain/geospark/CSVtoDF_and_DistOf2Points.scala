package com.oppchain.geospark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.datasyslab.geospark.serde.GeoSparkKryoRegistrator
import org.datasyslab.geosparksql.utils.GeoSparkSQLRegistrator

object CSVtoDF_and_DistOf2Points { //CSV_to_SpatialRDD
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("CSV-to-SpatialRDD").setMaster("local[*]")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val sparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("CSV-to-SpatialRDD")
      .config("spark.serializer", classOf[KryoSerializer].getName)
      .config("spark.kryo.registrator", classOf[GeoSparkKryoRegistrator].getName)
      .getOrCreate()

    // register GeoSpark User Defined Type, User Defined Function and optimized join query strategy
    GeoSparkSQLRegistrator.registerAll(sparkSession)

    // ------------------
    // create DF from CSV
    // ------------------
    val rawDf = sparkSession.read.format("csv")
               .option("delimiter", ",")
               .option("header", "true")
               .load("/Users/xiaoxiaosu/Documents/Codes/sample-data/NYC-taxi/MapZone/taxi_zones.csv")
    println("-- schema of DF ---")
    rawDf.printSchema()
    rawDf.createOrReplaceTempView("rawdf")
    println("-- DF ---")
    rawDf.show()

    // -----------------------
    // distance of 2 points
    // -----------------------

    // massy-rungis degree based dist
    val test_dist = sparkSession.sql(
      """
        |SELECT ST_Distance(ST_PointFromText('48.723884, 2.261092', ','),
        |                   ST_PointFromText('48.750687, 2.347233', ',')) AS distance
        |                   FROM rawdf
        """.stripMargin)
    test_dist.show(1)

    // massy-rungis meter based dist
    val test_dist_meter = sparkSession.sql(
      """
        |SELECT ST_Distance(ST_Transform(ST_PointFromText('48.723884, 2.261092', ','), "epsg:4326", "epsg:3857"),
        |                   ST_Transform(ST_PointFromText('48.750687, 2.347233', ','), "epsg:4326", "epsg:3857")) AS distance
        |                   FROM rawdf
        """.stripMargin)
    println("from massy to rungis, there is about 10km:: ")
    test_dist_meter.show(1)

    // massy-paris meter based dist
    val test_dist_meter_paris = sparkSession.sql(
      """
        |SELECT ST_Distance(ST_Transform(ST_PointFromText('48.723884, 2.261092', ','), "epsg:4326", "epsg:3857"),
        |                   ST_Transform(ST_PointFromText('48.875318, 2.302306', ','), "epsg:4326", "epsg:3857")) AS distance
        |                   FROM rawdf
        """.stripMargin)
    println("from massy to paris, there is about 25km:: ")
    test_dist_meter_paris.show(1)

    // massy - piriac meter based dist
    val test_dist_meter_loire = sparkSession.sql(
      """
        |SELECT ST_Distance(ST_Transform(ST_PointFromText('48.723884, 2.261092', ','), "epsg:4326", "epsg:3857"),
        |                   ST_Transform(ST_PointFromText('47.393880, -2.490085', ','), "epsg:4326", "epsg:3857")) AS distance
        |                   FROM rawdf
        """.stripMargin)
    println("from piriac sur mer to massy, there is about 500km: ")
    test_dist_meter_loire.show(1)
  }
}
