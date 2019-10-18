package PropertySale

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType, TimestampType}

object FileToHive extends App{

  Logger.getLogger("org").setLevel(Level.ERROR)

  // même si on utilise pas 'sc', on doit define, sinon le script ne marche pas
  val conf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
  val sc = new SparkContext(conf)

  val spark = SparkSession.builder().master("yarn").enableHiveSupport().appName("write File to Hive").getOrCreate()

  val path = "data_source/AlleghenyPropertySale.csv"

  val customSchema = StructType(Seq(
    StructField("ParId",StringType,true),
    StructField("HouseNum",IntegerType,true),
    StructField("Fraction",StringType,true),
    StructField("AddressDir",StringType,true),
    StructField("Street",StringType,true),
    StructField("AddressSuf",StringType,true),
    StructField("UnitDesc",StringType,true),
    StructField("UnitNo",StringType,true),
    StructField("City",StringType,true),
    StructField("State",StringType,true),
    StructField("ZipCode",StringType,true),
    StructField("SchoolCode",IntegerType,true),
    StructField("SchoolDesc",StringType,true),
    StructField("MuniCode",IntegerType,true),
    StructField("MuniDesc",StringType,true),
    StructField("RecodeDate",TimestampType,true),
    StructField("SaleDate",TimestampType,true),
    StructField("Price",IntegerType,true),
    StructField("DeedBook",IntegerType,true),
    StructField("DeedPage",IntegerType,true),
    StructField("SaleCode",StringType,true),
    StructField("SaleDesc",StringType,true),
    StructField("Instrtyp",StringType,true),
    StructField("InstrtypDesc",StringType,true)
  )
  )

  val PropertySaleDF = spark.read.schema(customSchema).option("header", "true").option("delimiter", ",").csv(path)

  PropertySaleDF.write.saveAsTable("PropertySale")
  // ça ne marche pas, parce que ce spark est local, et le hive est en gcloud. Ils sont pas dans le même reseaux et cluster

  spark.stop()
  sc.stop()

}
