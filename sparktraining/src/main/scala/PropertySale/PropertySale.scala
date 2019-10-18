package PropertySale

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

object PropertySale {

  def main(args: Array[String]) = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    // même si on utilise pas 'sc', on doit define, sinon le script ne marche pas
    val conf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val spark = SparkSession.builder().appName("PropertySale").config("spark.some.config","somevalue").getOrCreate()

    val path = "data_source/AlleghenyPropertySale.csv"

    val PropertySale = spark.read.option("header", "true").option("delimiter", ",").csv(path)

//    System.out.println("=== Print out schema ===")
//    PropertySale.printSchema()

    val DFSchema = PropertySale.schema
    println("DataFrame schema : ")
    println(DFSchema)

    val colNames = PropertySale.columns
    println("Columns are :")
    println(colNames.mkString(", "))

    val colType = PropertySale.dtypes
    println("PropertySale's columns name and type: ")
    colType.foreach(println)

    // meme col type en string, le summary get the real type
    println("Column summary: PRICE  ")
    val colDescription = PropertySale.describe("PRICE")
    colDescription.show()

    println("Column summary: SALEDATE  ")
    val colDescription2 = PropertySale.describe("SALEDATE")
    colDescription2.show() 

    /*System.out.println("=== Number of properties sold by city ===")
    PropertySale.groupBy("PROPERTYCITY").count().orderBy(desc("count")).show()

    System.out.println("=== [Multi Filtering] Valid Sale ===")
    //PropertySale.filter("SaleCode == '0' or SaleCode == 'U' or SaleCode == 'UR'").select("PROPERTYCITY","SCHOOLDESC","SALEDESC","SALECODE","PRICE").show()
    PropertySale.filter("SaleCode in ('0', 'U', 'UR')").select("PROPERTYCITY","SCHOOLDESC","SALEDESC","SALECODE","PRICE").show()

    // type DataFrame
    val PropertySale_typed = PropertySale.select(
      PropertySale.col("PROPERTYCITY"),
      PropertySale.col("SCHOOLDESC"),
      PropertySale.col("PRICE").cast("Integer"),
      PropertySale.col("SALEDATE").cast("Timestamp"),
      PropertySale.col("SALECODE")
    )
    System.out.println("=== schema of typed data frame ===")
    PropertySale_typed.printSchema()
*/

    spark.stop()
    sc.stop()
  }

}
