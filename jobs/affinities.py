"""
affinities.py
=============
Pipeline Step 6: User–Metadata Affinity Computation

Measures how strongly each user favours specific listing metadata values
(e.g. a particular neighbourhood or price bin) relative to their RFM cohort.

Affinity calculation:
  1. For every user booking up to their current frequency, join with listing
     metadata to determine which metadata values the user has engaged with.
  2. Compute a recency-adjusted frequency (RAdjF) per user per metadata
     value, where recent bookings contribute more.
  3. Min-max normalise RAdjF within each (RFM_Cohort, frequency, metadata)
     group to produce an affinity score in [0, 1].

This allows the model to capture that a user who disproportionately books
listings in a specific neighbourhood (compared to their cohort peers) has
high affinity for that neighbourhood.

Inputs:
  - data/{training,scoring}/cohorts_parquet      (from cohorts.py)
  - data/transactions_parquet                    (from simulate_txns.py)
  - data/listing_metadata_parquet                (from metadata.py)

Outputs:
  - data/{training,scoring}/affinities_parquet
"""

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window

spark = SparkSession.builder.master("local[4]").appName("Affinity").getOrCreate()

data_sets=['training','scoring']
for ds in data_sets:
    # -----------------------------------------------------------------
    # 1. Load cohort-enriched labels and transaction/metadata lookups
    # -----------------------------------------------------------------
    cohorts=spark.read.parquet(f'data/{ds}/cohorts_parquet')\
        .select(
            'RFM_Cohort'
            ,'user_id'
            ,'listing_id'
            ,'label'
            ,'frequency'
        )

    txns=spark.read.parquet('data/transactions_parquet')\
        .select(col('user_id').alias('txn_user_id')
        ,col('listing_id').alias('txn_listing_id')
        ,'booking_num'
        ,'days_since_last')

    max_days_since_last=txns.select(max("days_since_last")).collect()[0][0]

    # -----------------------------------------------------------------
    # 2. Reconstruct each user's booking history up to their current
    #    frequency and compute a recency adjustment factor (RAdj).
    #    RAdj = days_since_last / max_days so recent bookings ≈ 0 and
    #    older bookings approach 1.
    # -----------------------------------------------------------------
    running_txns=cohorts.join(txns,
        (txns.txn_user_id==cohorts.user_id)
        &(txns.booking_num<=cohorts.frequency)
        ,'inner'
    ).withColumn('RAdj',col('days_since_last')/max_days_since_last)

    # -----------------------------------------------------------------
    # 3. Join transaction history with listing metadata so each booking
    #    is annotated with all metadata facets (neighbourhood, price bin, etc.)
    # -----------------------------------------------------------------
    metadata=spark.read.parquet('data/listing_metadata_parquet')\
        .withColumnRenamed('listing_id','metadata_listing_id')
    running_txns_w_metadata=running_txns.join(metadata, 
        running_txns.txn_listing_id==metadata.metadata_listing_id,
        'left'
    )

    # -----------------------------------------------------------------
    # 4. Aggregate RAdjF per user per metadata value, then compute
    #    min/max per cohort group for normalisation
    # -----------------------------------------------------------------
    per_user_per_metadata=running_txns_w_metadata\
        .groupBy('RFM_Cohort','user_id','listing_id','label','frequency','metadata_type','metadata_value')\
        .agg(sum('RAdj').alias('RAdjF'))


    per_cohort_per_metadata=per_user_per_metadata\
        .groupBy('RFM_Cohort','frequency','metadata_type','metadata_value')\
        .agg(min('RAdjF').alias('min_RAdjF')
            ,max('RAdjF').alias('max_RAdjF')
        )

    # -----------------------------------------------------------------
    # 5. Compute the final affinity score via min-max normalisation
    #    within (cohort, frequency, metadata_type, metadata_value) groups.
    #    Score of 0 = lowest engagement in cohort; 1 = highest.
    # -----------------------------------------------------------------
    affinities=per_user_per_metadata.join(per_cohort_per_metadata
        , ['RFM_Cohort','frequency','metadata_type','metadata_value']
        , 'left'
    ).withColumn('affinity', when((col('max_RAdjF')-col('min_RAdjF'))==0,0.0).otherwise((col('RAdjF')-col('min_RAdjF'))/(col('max_RAdjF')-col('min_RAdjF'))))\
    .select('user_id','listing_id','label','frequency','metadata_type','metadata_value','RAdjF','affinity')

    affinities.write.mode('overwrite').parquet(f'data/{ds}/affinities_parquet')
