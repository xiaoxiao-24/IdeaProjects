package PropertySale

import java.sql.Timestamp

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType, TimestampType}

object PropertySale_DS_CaseClass extends App{

  val conf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
  val sc = new SparkContext(conf)

  val spark = SparkSession.builder().appName("PropertySale").getOrCreate()

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

  case class PropertySaleClass2(
                                ParId: String,
                                HouseNum: Integer,
                                Fraction: String,
                                AddressDir: String,
                                Street: String,
                                AddressSuf: String,
                                UnitDesc: String,
                                UnitNo: String,
                                City: String,
                                State: String,
                                ZipCode: String,
                                SchoolCode: Integer,
                                SchoolDesc: String,
                                MuniCode: Integer,
                                MuniDesc: String,
                                RecodeDate: Timestamp,
                                SaleDate: Timestamp,
                                Price: Integer,
                                DeedBook: Integer,
                                DeedPage: Integer,
                                SaleCode: String,
                                SaleDesc: String,
                                Instrtyp: String,
                                InstrtypDesc: String
                              )

  import spark.implicits._
  val PropertySaleDS = spark.read.schema(customSchema).option("header","true").option("delimiter",",").csv(path).as[PropertySaleClass2]

  PropertySaleDS.printSchema()

  spark.stop()
  sc.stop()

}
