import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window

spark = SparkSession.builder.master("local[4]").appName("Stitching").getOrCreate()

data_sets=['training','scoring']

for ds in data_sets:
    ones_and_zeros_rfm=spark.read.parquet(f'data/{ds}/ones_and_zeros_rfm_parquet')
    max_recency=ones_and_zeros_rfm.select(max("recency")).collect()[0][0]
    max_frequency=ones_and_zeros_rfm.select(max("frequency")).collect()[0][0]
    max_monetary=ones_and_zeros_rfm.select(max("monetary")).collect()[0][0]
    ones_and_zeros_rfm_bins=ones_and_zeros_rfm\
        .withColumn(
            "recency_bin",
            when(col("recency").isNull(), "null")
            .when((col("recency") >= 0) & (col("recency") < max_recency * 0.33), "high")  # 0 to 33
            .when((col("recency") >= max_recency * 0.33) & (col("recency") < max_recency * 0.66), "medium")  # 33 to 66
            .when((col("recency") >= max_recency * 0.66) & (col("recency") <= max_recency), "low")  # 66 to 100
            .otherwise("out_of_range")
        ).withColumn(
            "frequency_bin",
            when(col("frequency").isNull(), "null")
            .when((col("frequency") >= max_frequency * 0.80) & (col("frequency") <= max_frequency), "high")  # 66 to 100
            .when((col("frequency") >= max_frequency * 0.20) & (col("frequency") < max_frequency * 0.80), "medium")  # 33 to 66
            .when((col("frequency") >= 0) & (col("frequency") < max_frequency * 0.20), "low")  # 0 to 33
            .otherwise("out_of_range")
        ).withColumn(
            "monetary_bin",
            when(col("monetary").isNull(), "null")
            .when((col("monetary") >= max_monetary * 0.50) & (col("monetary") <= max_monetary), "high")  # 66 to 100
            .when((col("monetary") >= max_monetary * 0.15) & (col("monetary") < max_monetary * 0.50), "medium")  # 33 to 66
            .when((col("monetary") >= 0) & (col("monetary") < max_monetary * 0.15), "low")  # 0 to 33
            .otherwise("out_of_range")
        )

    col_list=['user_id','listing_id','label','frequency']

    user_rfm=ones_and_zeros_rfm_bins.select(*col_list
        ,lit('user_rfm').alias('feature_class')
        ,lit('recency').alias('feature_type')
        ,col('recency_bin').alias('feature_category')
        ,lit(1.0).alias('feature_value')
        ).union(
            ones_and_zeros_rfm_bins.select(*col_list
            ,lit('user_rfm').alias('feature_class')
            ,lit('frequency').alias('feature_type')
            ,col('frequency_bin').alias('feature_category')
            ,lit(1.0).alias('feature_value')
            )
        ).union(
            ones_and_zeros_rfm_bins.select(*col_list
            ,lit('user_rfm').alias('feature_class')
            ,lit('monetary').alias('feature_type')
            ,col('monetary_bin').alias('feature_category')
            ,lit(1.0).alias('feature_value')
            )
        )


    affinities=spark.read.parquet(f'data/{ds}/affinities_parquet')
    user_affinity=affinities.select(*col_list
        ,lit('user_affinities').alias('feature_class')
        ,col('metadata_type').alias('feature_type')
        ,col('metadata_value').alias('feature_category')
        ,col('affinity').alias('feature_value')
        )



    metadata=spark.read.parquet('data/listing_metadata_parquet')
    listing_features=metadata.select('listing_id'
        ,lit('listing').alias('feature_class')
        ,col('metadata_type').alias('feature_type')
        ,col('metadata_value').alias('feature_category')
        ,lit(1.0).alias('feature_value')
    )

    feature_values_labels=((ones_and_zeros_rfm.select(*col_list)
        .join(user_rfm, [*col_list],'inner'))
        .select(*col_list,'feature_class','feature_type','feature_category','feature_value')
        .union(
            ones_and_zeros_rfm.select(*col_list)
            .join(user_affinity, [*col_list],'inner')
            .select(*col_list,'feature_class','feature_type','feature_category','feature_value')
        )
        .union(
            ones_and_zeros_rfm.select(*col_list)
            .join(listing_features,'listing_id','inner')
            .select(*col_list,'feature_class','feature_type','feature_category','feature_value')
        )
    )

    feature_values_labels.write.mode('overwrite').parquet(f'data/{ds}/features_values_label_parquet')
    
    if ds=='training':
        feature_values_labels=spark.read.parquet('data/training/features_values_label_parquet')
        rowWindowSpec=Window.orderBy('feature_class','feature_type','feature_category')
        feature_values_labels\
            .select('feature_class','feature_type','feature_category')\
            .distinct()\
            .withColumn('feature_id',row_number().over(rowWindowSpec))\
            .write.mode('overwrite').csv('data/training/feature_dictionary_csv',header=True)


