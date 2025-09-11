import pandas as pd
import numpy as np
from sklearn.metrics import auc
from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window

from pyspark.ml.classification import FMClassifier, FMClassificationSummary, FMClassificationModel

### for training
row_identifier=spark.read.parquet('data/training/row_identifier_parquet')\
    .select('row_id','user_id','listing_id','frequency','label')

model_ready=spark.read.format('libsvm').load('data/training/libsvm')\
    .withColumn('row_id',col('label').cast('int'))\
    .drop('label')\
    .join(row_identifier,'row_id','inner')

weights = [0.9,0.1]
dfs=model_ready.randomSplit(weights,seed=2195)
training_df=dfs[0]
testing_df=dfs[1]

fs=10
reg=.05
fm = FMClassifier(factorSize=fs,regParam=reg)
fm.setSeed(2195)
model=fm.fit(training_df)
model.write().overwrite().save(f'models/factors_{fs}_reg_{reg}_model')

### for testinf
model=FMClassificationModel.load('models/factors_10_reg_0.05_model')

pr_curve=model.evaluate(testing_df).pr.toPandas().sort_values(by='recall')
auc_score = auc(pr_curve['recall'],pr_curve['precision'])
print(f"pr_auc: {auc_score * 100:.2f}%")


### for scoring
model=FMClassificationModel.load('models/factors_10_reg_0.05_model')

row_identifier=spark.read.parquet('data/scoring/row_identifier_parquet')\
    .select('row_id','user_id','listing_id','frequency')

scoring_df=spark.read.format('libsvm').load('data/scoring/libsvm')\
    .withColumn('row_id',col('label').cast('int'))\
    .drop('label')\
    .join(row_identifier,'row_id','inner')

windowSpec=Window.partitionBy('user_id').orderBy(desc('y_prob'))
positive_label=udf(lambda v: float(v[1]),DoubleType())
predictions=model.transform(scoring_df)\
    .withColumn('y_prob',positive_label('probability'))\
    .select('user_id','listing_id','y_prob')\
    .withColumn('rec_rank', row_number().over(windowSpec))

predictions.filter(col('rec_rank')<=10).show()
