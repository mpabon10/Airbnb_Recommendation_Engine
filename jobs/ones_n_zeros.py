import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window

zeros_mult=2

ones=spark.read.parquet('data/transactions_parquet').drop('price')\
    .withColumn('label',lit(1))\
    .filter(col('booking_num')>1)
    
users_df=spark.read.parquet('data/users_parquet').select('user_id')

listing_schema = StructType([
    StructField('listing_id', IntegerType(), True),
    StructField('host_is_superhost', IntegerType(), True),
    StructField('neighbourhood_group_cleansed', StringType(), True),
    StructField('accommodates', IntegerType(), True),
    StructField('price', DoubleType(), True),
    StructField('review_scores_rating', DoubleType(), True)
])

listing_df=spark.read.csv('data/nyc_listing_data_5pcnt_sample.csv',header=True, schema=listing_schema).select('listing_id')

user_listings=users_df.crossJoin(listing_df)

zeros=user_listings.join(ones.select('user_id','listing_id'),['user_id','listing_id'], 'left_anti')\
    .withColumn('label',lit(0))

num_ones=ones.groupBy('user_id').agg(count('*').alias('num_ones'))

window_spec = Window.partitionBy('user_id').orderBy('rand_num')
zeros_sample=zeros\
    .join(num_ones,'user_id','inner')\
    .withColumn('rand_num',rand(seed=2195))\
    .withColumn('row_num',row_number().over(window_spec))\
    .filter(col('row_num')<=col('num_ones')*zeros_mult)\
    .drop('rand_num','row_num','num_ones')

zeros_sample.persist()

zeros_sample.filter(col('user_id')==964).show()

#get random RFM from ones table to join to zeros table
zeros_sample_rfm=zeros_sample.drop('num_ones').join(num_ones,'user_id')\
    .withColumn('rand_num',rand(seed=2195))\
    .withColumn('booking_num',round((col('rand_num') * (col('num_ones') - 1)) + 2,0).cast('int'))\
    .join(ones.select('user_id','booking_num','days_since_last','running_avg_price')\
        ,['user_id','booking_num'],'left')

zeros_sample_rfm.persist()

output_cols=['user_id','listing_id','label',col('days_since_last').alias('recency'),(col('booking_num')-1).alias('frequency'),col('running_avg_price').alias('monetary')]

ones_and_zeros_rfm=ones.select(*output_cols)\
    .union(
      zeros_sample_rfm.select(*output_cols)
    )
    #.filter(col('frequency')>1)


ones_and_zeros_rfm.write.mode('overwrite').parquet('data/training/ones_and_zeros_rfm_parquet')
zeros_sample_rfm.unpersist()
user_rfm.unpersist()
zeros_sample.unpersist()

## For Scoring
last_booking=spark.read.parquet('data/transactions_parquet')\
    .groupBy('user_id').agg(max('booking_num').alias('last_booking'))

user_rfm=spark.read.parquet('data/transactions_parquet')\
    .join(last_booking,'user_id','left')\
    .filter(col('last_booking')==col('booking_num'))\
    .drop('listing_id')\
    .withColumn('running_avg_price',((col('running_avg_price')*col('last_booking'))+col('price'))/(col('last_booking')+1))\
    .filter(col('booking_num')>1)

num_candidates=100
windowSpec=Window.partitionBy('user_id').orderBy(rand())
listing_df=spark.read.csv('data/nyc_listing_data_5pcnt_sample.csv',header=True, schema=listing_schema).select('listing_id')

ones_and_zeros_frm_scoring=user_rfm.crossJoin(listing_df)\
    .withColumn('row_num', row_number().over(windowSpec))\
    .filter(col('row_num')<num_candidates)\
    .select('user_id','listing_id',lit(0).alias('label'),col('days_since_last').alias('recency'),col('booking_num').alias('frequency'),col('running_avg_price').alias('monetary'))\

ones_and_zeros_frm_scoring.write.mode('overwrite').parquet('data/scoring/ones_and_zeros_rfm_parquet')
