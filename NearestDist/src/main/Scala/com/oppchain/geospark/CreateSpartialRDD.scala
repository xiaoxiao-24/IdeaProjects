package com.oppchain.geospark

import com.vividsolutions.jts.geom.{Coordinate, GeometryFactory}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.datasyslab.geospark.formatMapper.shapefileParser.ShapefileReader
import org.datasyslab.geospark.serde.GeoSparkKryoRegistrator
import org.datasyslab.geospark.spatialOperator.KNNQuery
import org.datasyslab.geosparksql.utils.{Adapter, GeoSparkSQLRegistrator}

object CreateSpartialRDD {
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("CreateSpartialRDD").setMaster("local[*]")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val sparkSession = SparkSession.builder()
      .master("local[*]") // Delete this if run in cluster mode
      .appName("CreateSpartialRDD")
      // Enable GeoSpark custom Kryo serializer
      .config("spark.serializer", classOf[KryoSerializer].getName)
      .config("spark.kryo.registrator", classOf[GeoSparkKryoRegistrator].getName)
      .getOrCreate()

    // register GeoSpark User Defined Type, User Defined Function and optimized join query strategy
    GeoSparkSQLRegistrator.registerAll(sparkSession)

    // -----------------------------------------------------------
    // create RDD/DF from Shapefile
    // -----------------------------------------------------------

    val shapefileInputLocation = "/Users/xiaoxiaosu/Documents/Codes/sample-data/NYC-taxi/MapZone/NYC_TaxiZonesShapeFile"
    //val spatialRDD_Shapefile = ShapefileReader.readToGeometryRDD(sparkSession.sparkContext, shapefileInputLocation)
    val spatialRDD_Shapefile = ShapefileReader.readToPolygonRDD(sparkSession.sparkContext, shapefileInputLocation)
    val spatialDfShapefile = Adapter.toDf(spatialRDD_Shapefile, sparkSession)
    println("--- Schema of spatialDf from Shapefile ---")
    spatialDfShapefile.printSchema()
    println("--- Top 5 lines of spatialDfShapefile ---")
    spatialDfShapefile.show(5, false)

    // Create a Geometry type column
    spatialDfShapefile.createOrReplaceTempView("rawdf")
    val spatialDf = sparkSession.sql(
        """
          |SELECT ST_GeomFromWKT(geometry) AS col_geometry, borough,
          |       location_i, objectid, shape_area, shape_leng, zone
          |FROM rawdf
        """.stripMargin)
    println("--- Schema of spatialDF ---")
    spatialDf.printSchema()
    spatialDf.createOrReplaceTempView("spatialdf")
    println("--- Top 5 lines of spatialDf in degree ---")
    spatialDf.show(5, false)

    /*

    // Transform the Coordinate Reference System
    // from degree-based-CRS to meter-based-CRS: EPSG4326 -> EPSG3857
    val spatialDf_Coordinate = sparkSession.sql(
        """
          |SELECT ST_Transform(col_geometry, "epsg:4326", "epsg:3857") AS newshape, borough,
          |       location_i, objectid, shape_area, shape_leng, zone
          |FROM spatialdf
        """.stripMargin)
    println("--- Schema of spatialDf_Coordinate ---")
    spatialDf_Coordinate.printSchema()
    spatialDf_Coordinate.createOrReplaceTempView("spatialdfcoordinate")
    println("--- Top 5 lines of spatialDf in meters ---")
    spatialDf_Coordinate.show(20)

    // KNN: returns the 5 nearest neighbor of the given polygon
    val spatialDf_5knn = sparkSession.sql(
        """
          |SELECT borough, location_i, zone, ST_Distance(ST_PolygonFromEnvelope(1.0,100.0,1000.0,1100.0), newshape) AS distance
          |FROM spatialdfcoordinate
          |ORDER BY distance
          |LIMIT 5
        """.stripMargin)
    println("--- Schema of spatialDf_5knn ---")
    spatialDf_5knn.printSchema()
    spatialDf_5knn.createOrReplaceTempView("spatialdf5knn")
    println("--- Top 5 lines of spatialDf_5knn ---")
    spatialDf_5knn.show(5)


    // -----------------------------------------------------------
    // 2 ways to get 5 knn (k-nearest-neighbour) of a known point
    // -----------------------------------------------------------

    // known point: (40.206383, -75.090621)
    // zones to compare: 5 districts of NYC from a reference file

    // 1) from RDD

    // create a point with coordinate
    val geometryFactory = new GeometryFactory()
    val pointObject = geometryFactory.createPoint(new Coordinate(40.206383, -75.090621))
    println("--- "+pointObject+"'s polygon is ---")
    println(pointObject)

    // KNN: returns the 5 nearest neighbor of the given point with spatialOperator and RDD
    val K = 5 // K Nearest Neighbors
    val usingIndex = false
    val result = KNNQuery.SpatialKnnQuery(spatialRDD_Shapefile, pointObject, K, usingIndex)
    println("--- result of 5 knn of point('40.206383, -75.090621') by knnquery" + pointObject + " ---")
    println(result)

    // 2) from DataFrame + GeoSparkSQL

    // KNN: returns the 5 nearest neighbor of the given point with GeoSparkSQL and DF
    val spatialDf_5knn_point = sparkSession.sql(
      """
        |SELECT borough, location_i, zone,
        |       ST_Distance(ST_Transform(ST_PointFromText('40.206383, -75.090621', ','), "epsg:4326", "epsg:3857"), col_geometry) AS distance
        |FROM spatialdf
        |ORDER BY distance
        |LIMIT 5
        """.stripMargin)
    println("--- Schema of spatialDf_5knn_point ---")
    spatialDf_5knn_point.printSchema()
    spatialDf_5knn_point.createOrReplaceTempView("spatialdf5knnpoint")
    println("--- Top 5 knn of point('40.206383, -75.090621') by spatialDf_5knn_point ---")
    spatialDf_5knn_point.show(5)


     */
    sparkSession.stop()
  }
}
