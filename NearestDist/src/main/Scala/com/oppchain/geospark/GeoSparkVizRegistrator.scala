package com.oppchain.geospark

import org.apache.spark.sql.{SQLContext, SparkSession}
import org.datasyslab.geosparkviz.sql.UDF.UdfRegistrator
import org.datasyslab.geosparkviz.sql.UDT.UdtRegistrator

object GeoSparkVizRegistrator {
  def registerAll(sqlContext: SQLContext): Unit = {
    UdtRegistrator.registerAll()
    UdfRegistrator.registerAll(sqlContext)
  }

  def registerAll(sparkSession: SparkSession): Unit = {
    UdtRegistrator.registerAll()
    UdfRegistrator.registerAll(sparkSession)
  }

  def dropAll(sparkSession: SparkSession): Unit = {
    UdfRegistrator.dropAll(sparkSession)
  }
}
