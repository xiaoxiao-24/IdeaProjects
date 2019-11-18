package neumann


/*object imsi_svm extends App{

  val conf = new SparkConf().setAppName("SparkIMSI").setMaster("local[*]")
  val sc = new SparkContext(conf)

  Logger.getLogger("org").setLevel(Level.ERROR)

  val spark = SparkSession.builder().appName("IMSI").getOrCreate()

  val path = "data_source/imsi_json.json"

  val imsiDF = spark.read.json(path)

  imsiDF.createOrReplaceTempView("imsiDF")

  implicit def bool2int(b:Boolean) = if (b) 1 else 0
  spark.udf.register("bool2int", bool2int _)

  val imsi_train = spark.sql("""select arpu,averagemonthlybill, totalspent,
                              bool2int(smartphone), handsettype, category,type, daysactive,
                              dayslastactive, bool2int(canceled)
                              from imsiDF
                              where canceled is not null
                              and category <> ''
                              and type <> ''
                              and handsettype <> ''  """)

  // one-hot encoding on our categorical features
  // First we have to use the StringIndexer to convert the strings to integers.
  val indexer1 = new StringIndexer().
    setInputCol("handsettype").
    setOutputCol("handsettypeIndex").
    setHandleInvalid("keep")
  val DF_Churn1 = indexer1.fit(imsi_train).transform(imsi_train)

  val indexer2 = new StringIndexer().
    setInputCol("category").
    setOutputCol("categoryIndex").
    setHandleInvalid("keep")
  val DF_Churn2 = indexer2.fit(DF_Churn1).transform(DF_Churn1)

  val indexer3 = new StringIndexer().
    setInputCol("type").
    setOutputCol("typeIndex").
    setHandleInvalid("keep")
  val DF_Churn3 = indexer3.fit(DF_Churn2).transform(DF_Churn2)

  // use the OneHotEncoderEstimator to do the encoding.
  val encoder = new OneHotEncoderEstimator().
    setInputCols(Array("handsettypeIndex", "categoryIndex","typeIndex")).
    setOutputCols(Array("handsettypeVec", "categoryVec","typeVec"))
  val DF_Churn_encoded = encoder.fit(DF_Churn3).transform(DF_Churn3)

  // Spark models need exactly two columns: “label” and “features”
  // create label column
  val get_label = (DF_Churn_encoded.select(DF_Churn_encoded.col("UDF:bool2int(canceled)").as("label"),
    DF_Churn_encoded.col("arpu"), DF_Churn_encoded.col("averagemonthlybill"),
    DF_Churn_encoded.col("totalspent"), DF_Churn_encoded.col("UDF:bool2int(smartphone)"),
    DF_Churn_encoded.col("daysactive"), DF_Churn_encoded.col("dayslastactive"),
    DF_Churn_encoded.col("handsettypeVec"), DF_Churn_encoded.col("categoryVec"),
    DF_Churn_encoded.col("typeVec")))

  // assembler tous les features
  val assembler = new VectorAssembler().setInputCols(Array("arpu",
    "averagemonthlybill", "totalspent", "UDF:bool2int(smartphone)", "daysactive",
    "dayslastactive", "handsettypeVec", "categoryVec", "typeVec")).
    setOutputCol("features")

  // Transform the DataFrame
  val output = assembler.transform(get_label).select("label","features")

  //prepare dataset ( one part for train 70%, one for test 30%)
  // Splitting the data by create an array of the training and test data
  val Array(training, test) = output.select("label","features").
    randomSplit(Array(0.7, 0.3), seed = 12345)

  // create the training model
  val rf = new SVMWithSGD()

  // create the param grid
  val paramGrid = new ParamGridBuilder().
    addGrid(rf.numTrees,Array(20,50,100)).
    build()

}*/
