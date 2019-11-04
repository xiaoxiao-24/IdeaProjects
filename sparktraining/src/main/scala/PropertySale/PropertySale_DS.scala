package PropertySale

// Data source:
//        Allegheny County Property Sale Transactions
// link:  https://catalog.data.gov/dataset/allegheny-county-property-sale-transactions


import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._

object PropertySale_DS {

  def main(args: Array[String]) = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    // même si on utilise pas 'sc', on doit define, sinon le script ne marche pas
    val conf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val spark = SparkSession.builder().appName("PropertySale").config("spark.some.config","somevalue").getOrCreate()

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

    import spark.implicits._
    val PropertySaleDS = spark.read.schema(customSchema).option("header","true").option("delimiter",",").csv(path).as[PropertySaleClass]

    System.out.println("=== City Price ===")
    val CityDS = PropertySaleDS.select("ParId", "City", "Price").as[CityPrice]
    CityDS.printSchema()
    CityDS.show()

    System.out.println("=== Print out schema ===")
    PropertySaleDS.printSchema()

    System.out.println("=== City List ===")
    PropertySaleDS.select("City").distinct().show(false)

    println("=== Deal from Pittsburgh  ===")
    PropertySaleDS.where("city = 'PITTSBURGH'").show(false)

    System.out.println("=== Sale in City : PITTSBURGH ===")
    PropertySaleDS.filter(row => row.City == "PITTSBURGH").show(false)

    System.out.println("=== Number of sale descendant by school district ===")
    PropertySaleDS.groupBy("SchoolDesc").count().orderBy(desc("count")).show(false)

    System.out.println("=== Valid sales ===")
    PropertySaleDS.filter(row => row.SaleCode == "0" | row.SaleCode == "U" | row.SaleCode == "UR").show(false)

    System.out.println("=== Top 20 most expensive sale ===")
    PropertySaleDS.filter(row => row.SaleCode == "0" | row.SaleCode == "U" | row.SaleCode == "UR").orderBy(PropertySaleDS.col("Price").desc).show(false)

    //PropertySaleDS.orderBy(desc("Price")).show(false)

    System.out.println("=== Top 20 School district (number of sales) ===")
    PropertySaleDS.filter(row => row.SaleCode == "0" | row.SaleCode == "U" | row.SaleCode == "UR").groupBy("SchoolDesc").count().orderBy(desc("count")).show(false)

    System.out.println("=== Top 20 most expensive School district ===")
    PropertySaleDS.filter(row => row.SaleCode == "0" | row.SaleCode == "U" | row.SaleCode == "UR").groupBy("SchoolDesc").avg("Price").orderBy(desc("avg(Price)")).show(false)

    // create temprary view
    PropertySaleDS.createOrReplaceTempView("PropertySaleView")

    // Sale validation by type
    System.out.println("=== Distribution by sale's type (valid,no-valid) ===")
    spark.sql("select case when SaleCode = 'AA' then 'Undetermined' when SaleCode in ('0','U','UR') then 'ValidSale' when SaleCode in ('1','32','2','GV','3','6','9','13','14','16','19','27','33','34','35','36','37','99','BK','DT','H','N','PA') then 'InvalideSale' else 'DiscontinuedCodes' end as Type, count(1) from PropertySaleView group by Type").show(false)


    spark.stop()
  }

}
