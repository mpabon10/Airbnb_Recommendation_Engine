"""
ones_n_zeros.py
===============
Pipeline Step 4: Labelled Training & Scoring Data Creation

Builds the positive (label=1) and negative (label=0) example datasets for
the recommendation model.

**Training set:**
  - Positives ("ones"): actual user–listing bookings from transactions,
    enriched with RFM features (Recency, Frequency, Monetary).
  - Negatives ("zeros"): randomly sampled user–listing pairs the user
    did NOT book.  Negative samples are drawn at a 2× ratio relative to
    positives per user, and each is assigned a random RFM context from
    the user's real booking history.

**Scoring set:**
  - For every user, up to 100 random candidate listings are paired with
    the user's latest RFM snapshot.  All are labelled 0 (unknown) and
    will be scored by the model to produce recommendations.

Inputs:
  - data/transactions_parquet     (from simulate_txns.py)
  - data/users_parquet            (from simulate_txns.py)
  - data/nyc_listing_data_5pcnt_sample.csv

Outputs:
  - data/training/ones_and_zeros_rfm_parquet
  - data/scoring/ones_and_zeros_rfm_parquet
"""

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window

spark = SparkSession.builder.master("local[4]").appName("OnesNZeros").getOrCreate()

# ---------------------------------------------------------------------------
# 1. Build the POSITIVE examples (label = 1)
#    Each actual booking beyond the first is a positive training signal.
#    Booking #1 is excluded because there is no prior history to form
#    a meaningful RFM snapshot.
# ---------------------------------------------------------------------------
zeros_mult=2

ones=spark.read.parquet('data/transactions_parquet').drop('price')\
    .withColumn('label',lit(1))\
    .filter(col('booking_num')>1)

# ---------------------------------------------------------------------------
# 2. Build the NEGATIVE examples (label = 0)
#    Cross-join users × listings, then anti-join to remove user-listing
#    pairs that already have a booking.  Sample 2× negatives per user.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. Assign random RFM context to each negative example
#    Each zero sample borrows the recency / running_avg_price from a
#    randomly selected booking in the user's actual history so the model
#    receives realistic contextual features even for negative pairs.
# ---------------------------------------------------------------------------
zeros_sample_rfm=zeros_sample.drop('num_ones').join(num_ones,'user_id')\
    .withColumn('rand_num',rand(seed=2195))\
    .withColumn('booking_num',round((col('rand_num') * (col('num_ones') - 1)) + 2,0).cast('int'))\
    .join(ones.select('user_id','booking_num','days_since_last','running_avg_price')\
        ,['user_id','booking_num'],'left')

zeros_sample_rfm.persist()

# ---------------------------------------------------------------------------
# 4. Union positives and negatives, rename to standard RFM column names,
#    and write the training set to Parquet.
# ---------------------------------------------------------------------------
output_cols=['user_id','listing_id','label',col('days_since_last').alias('recency'),(col('booking_num')-1).alias('frequency'),col('running_avg_price').alias('monetary')]

ones_and_zeros_rfm=ones.select(*output_cols)\
    .union(
      zeros_sample_rfm.select(*output_cols)
    )
    #.filter(col('frequency')>1)


ones_and_zeros_rfm.write.mode('overwrite').parquet('data/training/ones_and_zeros_rfm_parquet')
zeros_sample_rfm.unpersist()
zeros_sample.unpersist()

# ---------------------------------------------------------------------------
# 5. Build the SCORING candidate set
#    For each user, cross-join with all listings and randomly sample up to
#    100 candidates.  RFM features come from the user's most recent booking.
# ---------------------------------------------------------------------------
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
