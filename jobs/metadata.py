import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window

spark = SparkSession.builder.master('local[4]').appName('Metadata').getOrCreate()

listing_schema = StructType([
    StructField('listing_id', IntegerType(), True),
    StructField('host_is_superhost', IntegerType(), True),
    StructField('neighbourhood_group_cleansed', StringType(), True),
    StructField('accommodates', IntegerType(), True),
    StructField('price', DoubleType(), True),
    StructField('review_scores_rating', DoubleType(), True)
])

listing_df=spark.read.csv('data/nyc_listing_data_5pcnt_sample.csv',header=True, schema=listing_schema)\
    .withColumn(
        'price_bin',
        when(col('price').isNull(), 'null')
        .when((col('price') >= 0) & (col('price') <= 50.99), '0-50')
        .when((col('price') > 50.99) & (col('price') <= 100.99), '51-100')
        .when((col('price') > 100.99) & (col('price') <= 200.99), '101-200')
        .when((col('price') > 200.99) & (col('price') <= 300.99), '201-300')
        .when((col('price') > 300.99) & (col('price') <= 500.99), '301-500')
        .otherwise('500+')
    )\
    .withColumn(
        'rating_bin',
        when(col('review_scores_rating').isNull(), 'null')
        .when(col('review_scores_rating') < 2, '0-1')  # 0 to < 2
        .when((col('review_scores_rating') >= 2) & (col('review_scores_rating') < 4), '2-3')  # 2 to < 4
        .when((col('review_scores_rating') >= 4) & (col('review_scores_rating') < 4.4), '4-4.3')  # 4 to < 4.4
        .when((col('review_scores_rating') >= 4.4) & (col('review_scores_rating') < 4.9), '4.4-4.8')  # 4.4 to < 4.9
        .when((col('review_scores_rating') >= 4.9) & (col('review_scores_rating') <= 5), '4.9-5.0')  # 4.9 to 5
        .otherwise('out_of_range')
)

listing_df.show()

metadata=listing_df.select(
    'listing_id'
    , lit('host_is_superhost').alias('metadata_type')
    , col('host_is_superhost').alias('metadata_value')
).union(
    listing_df.select(
    'listing_id'
    , lit('neighbourhood').alias('metadata_type')
    , col('neighbourhood_group_cleansed').alias('metadata_value')
    )
).union(
    listing_df.select(
    'listing_id'
    , lit('accommodates').alias('metadata_type')
    , (when(col('accommodates')>10,'+10').otherwise(col('accommodates'))).alias('metadata_value')
    )
).union(
    listing_df.select(
    'listing_id'
    , lit('price_bin').alias('metadata_type')
    , col('price_bin').alias('metadata_value')
    )
).union(
    listing_df.select(
    'listing_id'
    , lit('rating_bin').alias('metadata_type')
    , col('rating_bin').alias('metadata_value')
    )
)

metadata.show()

metadata.write.mode('overwrite').parquet('data/listing_metadata_parquet')
