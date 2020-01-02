/*
  Overlay 2 SQL DF represented images
*/

package com.oppchain.geospark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.SparkSession
import org.datasyslab.geospark.formatMapper.shapefileParser.ShapefileReader
import org.datasyslab.geosparksql.utils.{Adapter, GeoSparkSQLRegistrator}
import org.datasyslab.geosparkviz.core.{ImageGenerator, ImageSerializableWrapper, RasterOverlayOperator}
import org.datasyslab.geosparkviz.core.Serde.GeoSparkVizKryoRegistrator
import org.datasyslab.geosparkviz.utils.ImageType

object OverlayImageDF {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Visualise").setMaster("local[*]")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    val sparkSession = SparkSession.builder()
      .master("local[*]") // Delete this if run in cluster mode
      .appName("Visualise")
      .config("spark.serializer", classOf[KryoSerializer].getName)
      .config("spark.kryo.registrator", classOf[GeoSparkVizKryoRegistrator].getName)
      .getOrCreate()

    GeoSparkSQLRegistrator.registerAll(sparkSession)
    GeoSparkVizRegistrator.registerAll(sparkSession)


    // --------------------------------------------
    // NYC Zone file: create RDD->DF from Shapefile
    // --------------------------------------------

    // Shapefile->RDD
    val shapefileInputLocation = "/Users/xiaoxiaosu/Documents/Codes/sample-data/NYC-taxi/MapZone/NYC_TaxiZonesShapeFile"
    val spatialRDD_Shapefile = ShapefileReader.readToGeometryRDD(sparkSession.sparkContext, shapefileInputLocation)

    // RDD->DF
    val spatialDfShapefile = Adapter.toDf(spatialRDD_Shapefile, sparkSession)

    // Create a Geometry type column
    spatialDfShapefile.createOrReplaceTempView("rawdf")
    val spatialDf = sparkSession.sql(
      """
        |SELECT ST_GeomFromWKT(geometry) AS col_geometry, borough, zone
        |FROM rawdf
        """.stripMargin)
    //println("--- Schema of spatialDF ---")
    //spatialDf.printSchema()
    spatialDf.createOrReplaceTempView("zone")
    //println("--- Top 5 lines of spatialDf in degree ---")
    //spatialDf.show(5)

    // Pixelize spatial objects: compute the spatial boundary of this column
    val zoneboundtable = sparkSession.sql(
      """
        |SELECT ST_Envelope_Aggr(col_geometry) as bound FROM zone
        |""".stripMargin
    )
    //println("--- schema of zone boundtable: boundtable ---")
    //zoneboundtable.printSchema()
    //zoneboundtable.show(5, false)
    zoneboundtable.createOrReplaceTempView("zone_boundtable")


    // use ST_Pixelize to conver them to pixels.
    val zonepixels = sparkSession.sql(
      """
        |SELECT pixel, col_geometry
        |  FROM zone
        |LATERAL VIEW ST_Pixelize(ST_Transform(col_geometry, 'epsg:4326','epsg:3857'), 1000, 1000,
        |                            (SELECT ST_Transform(bound, 'epsg:4326','epsg:3857')
        |                               FROM zone_boundtable)) AS pixel
        |""".stripMargin
    )
    //println("--- schema of zone pixels ---")
    //zonepixels.printSchema()
    //zonepixels.show(5, false)
    zonepixels.createOrReplaceTempView("zone_pixels")

    // Aggregate pixels
    val zonepixelsAgg = sparkSession.sql(
      """
        |SELECT pixel, count(*) as weight
        |FROM zone_pixels
        |GROUP BY pixel
        |""".stripMargin
    )
    //println("--- schema of zone pixelsAgg: zone pixelaggregates ---")
    //zonepixelsAgg.printSchema()
    //zonepixelsAgg.show(5, false)
    zonepixelsAgg.createOrReplaceTempView("zone_pixelaggregates")

    /*
    sparkSession.sql(
      """
        |select count(*) from zone_pixelaggregates
        |""".stripMargin).show()

    //create tile name for every pixel
    val zonepixelsAgg_tile = sparkSession.sql(
      """
        |SELECT pixel, weight, ST_TileName(pixel, 3) AS pid
        |FROM zone_pixelaggregates
        |""".stripMargin
    )
    //println("--- schema of zone pixelsAgg tile: tile pixelaggregates ---")
    //zonepixelsAgg_tile.printSchema()
    //zonepixelsAgg_tile.show(5, false)
    zonepixelsAgg_tile.createOrReplaceTempView("tile_pixelaggregates")
*/
    // Colorize pixels
    // SELECT pixel, pid, ST_Colorize(weight, (select max(weight) from zone_pixelaggregates), 'blue') as color
    val zonepixelsColor = sparkSession.sql(
      """
        |SELECT pixel, ST_Colorize(weight, (select max(weight) from zone_pixelaggregates), 'blue') as color
        |FROM zone_pixelaggregates
        |""".stripMargin
    )
    //println("--- schema of zone pixelsAgg: zone_pixelcolor ---")
    //zonepixelsColor.printSchema()
    //zonepixelsColor.show(5, false)
    zonepixelsColor.createOrReplaceTempView("zone_pixelcolor")

    // group pixels by tiles and render map tile images
    //  GROUP BY pid
    val tilePlot = sparkSession.sql(
      """
        |SELECT ST_Render(pixel, color) AS image,
        |       (SELECT ST_AsText(bound) FROM zone_boundtable) AS boundary
        |  FROM zone_pixelcolor
        |""".stripMargin
    )
    //println("--- schema of zone: images ---")
    //tilePlot.printSchema()
    //tilePlot.show(5, false)
    tilePlot.createOrReplaceTempView("zone_images")

    // Store map tiles on disk
    // Fetch the image from the previous DataFrame
    var zone_image = sparkSession.table("zone_images").take(1)(0)(0).asInstanceOf[ImageSerializableWrapper].getImage

    // Use GeoSparkViz ImageGenerator to store this image on disk.
    //var imageGenerator = new ImageGenerator
    //imageGenerator.SaveRasterImageAsLocalFile(zone_image, System.getProperty("user.dir") + "/target/nyc_points", ImageType.PNG)


    // -----------------------------
    // Taxi file: create DF from CSV
    // -----------------------------
    val raw_nytaxi = sparkSession.read.option("header", "true").csv("/Users/xiaoxiaosu/Documents/Codes/sample-data/nyctaxisub.csv")
    println("--- the NYC taxi file schema ---")
    raw_nytaxi.printSchema()

    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.types._

    val toInt = udf[Int, String](_.toInt)
    val toDouble = udf[Double, String](_.toDouble)

    val nyTaxi = raw_nytaxi.withColumn("TripDistance", toDouble(raw_nytaxi("trip_distance")))
      .withColumn("TripTimeInSecs", toInt(raw_nytaxi("trip_time_in_secs")))
      .withColumn("Rate", toInt(raw_nytaxi("rate_code")))
      .withColumn("NbPassenger", toInt(raw_nytaxi("passenger_count")))
      .withColumn("DropOffLat", toDouble(raw_nytaxi("dropoff_latitude")))
      .withColumn("DropOffLong", toDouble(raw_nytaxi("dropoff_longitude")))
      .withColumn("PickUpLat", toDouble(raw_nytaxi("pickup_latitude")))
      .withColumn("PickUpLong", toDouble(raw_nytaxi("pickup_longitude")))
      .withColumn("DropOffTime", unix_timestamp(col("dropoff_datetime"), "yyyy-MM-dd HH:mm:ss").cast(TimestampType).as("timestamp"))
      .withColumn("PickUpTime", unix_timestamp(col("pickup_datetime"), "yyyy-MM-dd HH:mm:ss").cast(TimestampType).as("timestamp"))

    val nyTaxiDf = nyTaxi.select("_id", "_rev", "DropOffTime", "PickUpTime", "DropOffLat", "DropOffLong", "PickUpLat", "PickUpLong", "NbPassenger", "Rate")

    nyTaxiDf.createOrReplaceTempView("NycTaxiView")

    // Create Spatial DataFrame
    val NycTaxiDF = sparkSession.sql(
      """
        |SELECT ST_PointFromText(concat(DropOffLong, ',', DropOffLat), ',') as shape
        |FROM NycTaxiView
        |where DropOffLong between -74.257159 and -73.699215
        |  and DropOffLat between 40.495992 and 40.915568
        |""".stripMargin
    )
    //println("--- schema of NycTaxiDF: pointtable ---")
    //NycTaxiDF.printSchema()
    //NycTaxiDF.show(5, false)
    NycTaxiDF.createOrReplaceTempView("pointtable")

    // Pixelize spatial objects: compute the spatial boundary of this column
    val NycTaxiboundtable = sparkSession.sql(
      """
        |SELECT ST_Envelope_Aggr(shape) as bound FROM pointtable
        |""".stripMargin
    )
    //println("--- schema of NycTaxiboundtable: boundtable ---")
    //NycTaxiboundtable.printSchema()
    //NycTaxiboundtable.show(5, false)
    NycTaxiboundtable.createOrReplaceTempView("boundtable")

    // use ST_Pixelize to conver them to pixels.
    val NycTaxipixels = sparkSession.sql(
      """
        |SELECT pixel, shape
        |  FROM pointtable
        |LATERAL VIEW ST_Pixelize(ST_Transform(shape, 'epsg:4326','epsg:3857'), 1000, 1000,
        |                            (SELECT ST_Transform(bound, 'epsg:4326','epsg:3857')
        |                               FROM boundtable)) AS pixel
        |""".stripMargin
    )
    //println("--- schema of NycTaxipixels ---")
    //NycTaxipixels.printSchema()
    //NycTaxipixels.show(20, false)
    NycTaxipixels.createOrReplaceTempView("pixels")

    // Aggregate pixels
    val NycTaxipixelsAgg = sparkSession.sql(
      """
        |SELECT pixel, count(*) as weight
        |FROM pixels
        |GROUP BY pixel
        |""".stripMargin
    )
    //println("--- schema of NycTaxipixelsAgg: pixelaggregates ---")
    //NycTaxipixelsAgg.printSchema()
    //NycTaxipixelsAgg.show(20, false)
    NycTaxipixelsAgg.createOrReplaceTempView("pixelaggregates")

    // Colorize pixels
    val NycTaxipixelsColor = sparkSession.sql(
      """
        |SELECT pixel, ST_Colorize(weight, (SELECT max(weight) FROM pixelaggregates)) as color
        |FROM pixelaggregates
        |""".stripMargin
    )
    //println("--- schema of NycTaxipixelsAgg: pixelcolor ---")
    //NycTaxipixelsColor.printSchema()
   // NycTaxipixelsColor.show(5, false)
    NycTaxipixelsColor.createOrReplaceTempView("pixelcolor")

    // Render the image (plot)
    val NycTaxipixelsPlot = sparkSession.sql(
      """
        |SELECT ST_Render(pixel, color) AS image,
        |       (SELECT ST_AsText(bound) FROM boundtable) AS boundary
        |  FROM pixelcolor
        |""".stripMargin
    )
    //println("--- schema of NycTaxipixelsAgg: images ---")
    //NycTaxipixelsPlot.printSchema()
    //NycTaxipixelsPlot.show(5, false)
    NycTaxipixelsPlot.createOrReplaceTempView("images")

    // Store the image on disk
    // Fetch the image from the previous DataFrame
    var image = sparkSession.table("images").take(1)(0)(0).asInstanceOf[ImageSerializableWrapper].getImage

    // ----------------------
    // overlay 2 layer images
    // ----------------------
    val overlayOperator = new RasterOverlayOperator(image)
    overlayOperator.JoinImage(zone_image)

    // ----------------------
    // show overlayed image
    // ----------------------
    // Use GeoSparkViz ImageGenerator to store this image on disk.
    var imageGenerator = new ImageGenerator
    imageGenerator.SaveRasterImageAsLocalFile(overlayOperator.backRasterImage, System.getProperty("user.dir") + "/target/sql_coimage_2", ImageType.PNG)

    /*
        val imagestring = sparkSession.sql(
          """
            |SELECT ST_EncodeImage(image)
            |	 FROM images
            |""".stripMargin
        )
        imagestring.createOrReplaceTempView("imagestring")
        sparkSession.table("imagestring").show(false)
   */

    sparkSession.stop()
  }
}
