from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from etl import ETLPipeline
from model import *
import mlflow

def build_spark_session():
    builder = SparkSession.builder \
        .appName("lab3-app") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
    return configure_spark_with_delta_pip(builder).getOrCreate()

def main():
    spark = build_spark_session()
    mlflow.set_tracking_uri("http://mlflow:5000")

    data_dir = "data/"
    gold_path = "data/golden/data"

    etl = ETLPipeline(spark, data_dir)
    etl.transform()

    data, target, features = load_data(spark, gold_path)
    model = train(data, target)
    results = evaluate(model, data, target)
    
    logmlperf(model, results, features)

    spark.stop()

if __name__ == '__main__':
    main()