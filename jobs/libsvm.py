import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window

spark = SparkSession.builder.master("local[4]").appName("libsvm").getOrCreate()

feature_dictionary=spark.read.csv('data/training/feature_dictionary_csv',header=True)\
    .withColumn('feature_id',col('feature_id').cast('int'))

num_features_plus_one=feature_dictionary.count()+1

# Convert to LIBSVM format
# The added 0.0 at the end at index 1 plus num features ensures the
# sparse vectors generated from the libsvm strings in the training step are the same size
def to_libsvm_format(features):
    return " ".join(f"{f['feature_id']}:{f['feature_value']}" for f in features)+' '+str(num_features_plus_one)+':0.0'
# Register UDF to convert to LIBSVM format
to_libsvm_udf = udf(to_libsvm_format, StringType())

data_sets=['training','scoring']
for ds in data_sets:
    feature_values_labels=spark.read.parquet(f'data/{ds}/features_values_label_parquet')
    feature_values_label_w_id=feature_values_labels\
        .join(feature_dictionary,['feature_class','feature_type','feature_category'],'left')

    # Group By User and Collect Features
    libsvm_df = (feature_values_label_w_id
        .groupBy('user_id','listing_id','label','frequency')
        .agg(collect_list(struct('feature_id', 'feature_value')).alias('features'))
        .withColumn('sorted_features',sort_array('features'))
        .withColumn('libsvm_features', to_libsvm_udf('sorted_features'))
        .withColumn('row_id',row_number().over(Window.orderBy(col('user_id'))))
    )

    libsvm_df.write.mode('overwrite').parquet(f'data/{ds}/row_identifier_parquet')
    libsvm_df=spark.read.parquet(f'data/{ds}/row_identifier_parquet')
    # Format the final LIBSVM representation
    model_ready = (libsvm_df
    .select(
        concat_ws(' ', 'row_id', 'libsvm_features').alias('libsvm_format')
    )
                    )
    model_ready.write.mode('overwrite').text(f'data/{ds}/libsvm')