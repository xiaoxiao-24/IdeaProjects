def get_data_ready(df, categorical_cols, continuous_cols, label_col):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql.functions import col

    indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                for c in categorical_cols]

    # default setting: dropLast=True
    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(),
                              outputCol="{0}_encoded".format(indexer.getOutputCol()))
                for indexer in indexers]

    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders] + continuous_cols,
                                outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    model = pipeline.fit(df)
    data = model.transform(df)

    data = data.withColumn('label', col(label_col))

    return data.select('features', 'label')
