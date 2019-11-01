package PropertySale

/*
 *   SparkSQL with remote Hive
 */

/*object CatalogAPI_Hive extends App {

  Logger.getLogger("org").setLevel(Level.ERROR)

  // même si on utilise pas 'sc', on doit define, sinon le script ne marche pas
  val conf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")
  val sc = new SparkContext(conf)

  val warehouseLocation = new File("spark-warehouse").getAbsolutePath

  val spark = SparkSession.builder().master("yarn").config("spark.sql.warehouse.dir",warehouseLocation).enableHiveSupport().appName("catalog API to Hive").getOrCreate()

  val catalog = spark.catalog

  val schema = StructType(Seq(
    StructField("Name",StringType,true),
    StructField("Age",IntegerType,true))
  )

  val employee = spark.read.option("inferSchema",true).csv("/Users/xiaoxiaorey/Documents/Codes/sample-data/employee.csv")

  catalog.createTable("employee","csv",schema,Map("Comments" -> "table created with API"))

  println("Table created: " + catalog.tableExists("employee"))

  employee.write.insertInto("employee")

  spark.table("employee").show()

  spark.stop()
  sc.stop()

}*/
