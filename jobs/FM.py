"""
FM.py
=====
Pipeline Step 9: Factorization Machine Training, Evaluation & Scoring

Trains a Spark ML FMClassifier (Factorization Machine for binary
classification) on the LIBSVM-formatted training data, evaluates it using
the Precision–Recall AUC on a held-out test split, and then scores the
candidate listing set to produce personalised recommendations.

Hyperparameters:
  - factorSize = 10   (dimensionality of latent factor vectors)
  - regParam   = 0.05 (L2 regularisation strength)

The final output ranks candidate listings per user by predicted probability
of booking (y_prob) and displays the top-10 recommendations.

Inputs:
  - data/{training,scoring}/row_identifier_parquet  (from libsvm.py)
  - data/{training,scoring}/libsvm                  (from libsvm.py)

Outputs:
  - models/factors_10_reg_0.05_model
  - Top-10 recommendations per user (printed to console)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import auc
from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window

from pyspark.ml.classification import FMClassifier, FMClassificationSummary, FMClassificationModel

spark = SparkSession.builder.master("local[4]").appName("FM").getOrCreate()

# =========================================================================
# TRAINING
# =========================================================================
# Load row identifiers (user_id, listing_id, label) and LIBSVM feature data,
# then join them on row_id so each feature vector has its true label.
# =========================================================================
row_identifier=spark.read.parquet('data/training/row_identifier_parquet')\
    .select('row_id','user_id','listing_id','frequency','label')

model_ready=spark.read.format('libsvm').load('data/training/libsvm')\
    .withColumn('row_id',col('label').cast('int'))\
    .drop('label')\
    .join(row_identifier,'row_id','inner')

# 90/10 train-test split (deterministic seed for reproducibility)
weights = [0.9,0.1]
dfs=model_ready.randomSplit(weights,seed=2195)
training_df=dfs[0]
testing_df=dfs[1]

# Train the Factorization Machine classifier
fs=10
reg=.05
fm = FMClassifier(factorSize=fs,regParam=reg)
fm.setSeed(2195)
model=fm.fit(training_df)
model.write().overwrite().save(f'models/factors_{fs}_reg_{reg}_model')

# =========================================================================
# EVALUATION
# =========================================================================
# Compute the Precision–Recall AUC on the held-out test set.
# =========================================================================
model=FMClassificationModel.load('models/factors_10_reg_0.05_model')

pr_curve=model.evaluate(testing_df).pr.toPandas().sort_values(by='recall')
auc_score = auc(pr_curve['recall'],pr_curve['precision'])
print(f"pr_auc: {auc_score * 100:.2f}%")


# =========================================================================
# SCORING
# =========================================================================
# Score the candidate listing set and rank recommendations per user.
# =========================================================================
model=FMClassificationModel.load('models/factors_10_reg_0.05_model')

row_identifier=spark.read.parquet('data/scoring/row_identifier_parquet')\
    .select('row_id','user_id','listing_id','frequency')

scoring_df=spark.read.format('libsvm').load('data/scoring/libsvm')\
    .withColumn('row_id',col('label').cast('int'))\
    .drop('label')\
    .join(row_identifier,'row_id','inner')

# Extract the probability of the positive class and rank listings per user
windowSpec=Window.partitionBy('user_id').orderBy(desc('y_prob'))
positive_label=udf(lambda v: float(v[1]),DoubleType())
predictions=model.transform(scoring_df)\
    .withColumn('y_prob',positive_label('probability'))\
    .select('user_id','listing_id','y_prob')\
    .withColumn('rec_rank', row_number().over(windowSpec))

predictions.filter(col('rec_rank')<=10).show()
