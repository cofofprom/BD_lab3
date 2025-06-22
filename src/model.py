from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark

def load_data(spark, path):
    df = spark.read.format("delta").load(path)

    target = 'Diabetes_012'
    non_cat_cols = ['BMI_scaled', 'MentHlth_scaled', 'PhysHlth_scaled', 'Age_scaled', 'Income_scaled']
    cat_cols = [c for c in df.columns if c not in non_cat_cols and c != target]

    all_cols = cat_cols + non_cat_cols
    assembler = VectorAssembler(inputCols=all_cols, outputCol="features")
    df = assembler.transform(df)

    return df, target, all_cols

def train(df, target):
    model = LogisticRegression(featuresCol='features', labelCol=target)
    pipeline = Pipeline(stages=[model]).fit(df)
    return pipeline

def evaluate(model, df, target):
    df = model.transform(df)
    df.cache()

    metrics = ["f1", "accuracy", "weightedPrecision", "weightedRecall"]
    results = {}

    evaluator = MulticlassClassificationEvaluator(labelCol=target)
    for metric in metrics:
        evaluator.setMetricName(metric)
        results[metric] = evaluator.evaluate(df)

    df.unpersist()

    return results

def logmlperf(model, metrics, feature_cols):
    with mlflow.start_run():
        mlflow.spark.log_model(model, "logreg")

        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("feature_names", feature_cols)

        model_params = model.stages[-1].extractParamMap()
        for param, value in model_params.items():
            mlflow.log_param(param.name, value)