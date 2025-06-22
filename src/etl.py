import os.path as osp
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder
from delta.tables import DeltaTable

class ETLPipeline:
    def __init__(self, spark, data_dir):
        self.spark = spark
        self.raw_data = osp.join(data_dir, 'data.csv')
        self.bronze_path = osp.join(data_dir, 'bronze', 'data')
        self.silver_path = osp.join(data_dir, 'silver', 'data')
        self.golden_path = osp.join(data_dir, 'golden', 'data')

    def process_bronze(self):
        df = self.spark.read.csv(self.raw_data, header=True)
        df = df.repartition(self.spark.sparkContext.defaultParallelism)

        target = 'Diabetes_012'
        non_cat_cols = set(['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Income'])
        cat_cols = set([c for c in df.columns if c not in non_cat_cols and c != target])

        df = df.select(
            *[df[col_name].cast('int' if col_name in cat_cols or \
                                col_name == target \
                                else 'double') for col_name in df.columns]
        )
        
        df.write.format("delta").mode("overwrite").save(self.bronze_path)

    def process_silver(self):
        non_cat_cols = ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Income']
        delta_table = DeltaTable.forPath(self.spark, self.bronze_path)
        delta_table.optimize().executeZOrderBy(*non_cat_cols)
        df = self.spark.read.format("delta").load(self.bronze_path)
        df = df.repartition(self.spark.sparkContext.defaultParallelism)
        df = df.na.drop()

        for col_name in non_cat_cols:
            q05 = df.approxQuantile(col_name, [0.05], 0.25)[0]
            q95 = df.approxQuantile(col_name, [0.95], 0.25)[0]
            df = df.filter((F.col(col_name) >= q05) & (F.col(col_name) <= q95))

        df.write.format("delta").mode("overwrite").save(self.silver_path)

    def process_gold(self):
        df = self.spark.read.format("delta").load(self.silver_path)
        df = df.repartition(self.spark.sparkContext.defaultParallelism)
        target = 'Diabetes_012'
        non_cat_cols = set(['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Income'])
        cat_cols = set([c for c in df.columns if c not in non_cat_cols and c != target])

        agg_exprs = [
            F.mean(c).alias(f"{c}_mean") for c in non_cat_cols
        ] + [
            F.stddev(c).alias(f"{c}_std") for c in non_cat_cols
        ]

        stats = df.agg(*agg_exprs).first().asDict()

        for col_name in non_cat_cols:
            mean = stats[f"{col_name}_mean"]
            std = stats[f"{col_name}_std"]
            df = df.withColumn(f'{col_name}_scaled', (F.col(col_name) - mean) / std)
            df = df.drop(col_name)

        for col_name in cat_cols:
            encoder = OneHotEncoder(
                inputCol=f"{col_name}",
                outputCol=f"{col_name}_oh",
                dropLast=True,
            )
            df = encoder.fit(df).transform(df)
            df = df.drop(col_name)

        df.write.format("delta").mode("overwrite").save(self.golden_path)

    def transform(self):
        self.process_bronze()
        self.process_silver()
        self.process_gold()