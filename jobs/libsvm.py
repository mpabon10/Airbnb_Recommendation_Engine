"""
libsvm.py
=========
Pipeline Step 8: LIBSVM Format Conversion

Converts the assembled feature tables into LIBSVM-formatted text files that
can be consumed by Spark ML’s FMClassifier.

For each user–listing pair the script:
  1. Loads the feature dictionary (from feature_stitching.py) to map each
     (feature_class, feature_type, feature_category) to a numeric feature ID.
  2. Joins feature values with their IDs, groups by (user_id, listing_id),
     collects and sorts the sparse feature vector, and serialises it to
     the LIBSVM string format:  <row_id> <fid>:<value> <fid>:<value> ...
  3. Appends a sentinel feature (ID = num_features + 1, value = 0.0) to
     guarantee all sparse vectors have identical dimensionality after
     Spark ML parses them.

A row-identifier Parquet file is also written so downstream steps can map
row IDs back to (user_id, listing_id, label, frequency).

Inputs:
  - data/training/feature_dictionary_csv             (from feature_stitching.py)
  - data/{training,scoring}/features_values_label_parquet  (from feature_stitching.py)

Outputs:
  - data/{training,scoring}/row_identifier_parquet
  - data/{training,scoring}/libsvm
"""

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window

spark = SparkSession.builder.master("local[4]").appName("libsvm").getOrCreate()

# ---------------------------------------------------------------------------
# 1. Load the feature dictionary that maps feature names to integer IDs
# ---------------------------------------------------------------------------
feature_dictionary=spark.read.csv('data/training/feature_dictionary_csv',header=True)\
    .withColumn('feature_id',col('feature_id').cast('int'))

num_features_plus_one=feature_dictionary.count()+1

# ---------------------------------------------------------------------------
# 2. UDF to serialise a sorted list of (feature_id, feature_value) structs
#    into a single LIBSVM string.  The sentinel feature at the end ensures
#    consistent sparse-vector length.
# ---------------------------------------------------------------------------
def to_libsvm_format(features):
    return " ".join(f"{f['feature_id']}:{f['feature_value']}" for f in features)+' '+str(num_features_plus_one)+':0.0'

to_libsvm_udf = udf(to_libsvm_format, StringType())

# ---------------------------------------------------------------------------
# 3. Process both training and scoring datasets
# ---------------------------------------------------------------------------
data_sets=['training','scoring']
for ds in data_sets:
    # Load feature values and join with the feature dictionary to get IDs
    feature_values_labels=spark.read.parquet(f'data/{ds}/features_values_label_parquet')
    feature_values_label_w_id=feature_values_labels\
        .join(feature_dictionary,['feature_class','feature_type','feature_category'],'left')

    # Group features by (user, listing), sort by feature_id, and serialise
    libsvm_df = (feature_values_label_w_id
        .groupBy('user_id','listing_id','label','frequency')
        .agg(collect_list(struct('feature_id', 'feature_value')).alias('features'))
        .withColumn('sorted_features',sort_array('features'))
        .withColumn('libsvm_features', to_libsvm_udf('sorted_features'))
        .withColumn('row_id',row_number().over(Window.orderBy(col('user_id'))))
    )

    # Write a row-identifier Parquet for downstream model evaluation
    libsvm_df.write.mode('overwrite').parquet(f'data/{ds}/row_identifier_parquet')
    libsvm_df=spark.read.parquet(f'data/{ds}/row_identifier_parquet')

    # Write the final LIBSVM text file (row_id followed by sparse features)
    model_ready = (libsvm_df
    .select(
        concat_ws(' ', 'row_id', 'libsvm_features').alias('libsvm_format')
    )
                    )
    model_ready.write.mode('overwrite').text(f'data/{ds}/libsvm')