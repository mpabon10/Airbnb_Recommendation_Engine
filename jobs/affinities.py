import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window

spark = SparkSession.builder.master("local[4]").appName("Affinity").getOrCreate()

data_sets=['training','scoring']
for ds in data_sets:
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

    running_txns=cohorts.join(txns,
        (txns.txn_user_id==cohorts.user_id)
        &(txns.booking_num<=cohorts.frequency)
        ,'inner'
    ).withColumn('RAdj',col('days_since_last')/max_days_since_last)

    metadata=spark.read.parquet('data/listing_metadata_parquet')\
        .withColumnRenamed('listing_id','metadata_listing_id')
    running_txns_w_metadata=running_txns.join(metadata, 
        running_txns.txn_listing_id==metadata.metadata_listing_id,
        'left'
    )

    per_user_per_metadata=running_txns_w_metadata\
        .groupBy('RFM_Cohort','user_id','listing_id','label','frequency','metadata_type','metadata_value')\
        .agg(sum('RAdj').alias('RAdjF'))


    per_cohort_per_metadata=per_user_per_metadata\
        .groupBy('RFM_Cohort','frequency','metadata_type','metadata_value')\
        .agg(min('RAdjF').alias('min_RAdjF')
            ,max('RAdjF').alias('max_RAdjF')
        )

    affinities=per_user_per_metadata.join(per_cohort_per_metadata
        , ['RFM_Cohort','frequency','metadata_type','metadata_value']
        , 'left'
    ).withColumn('affinity', when((col('max_RAdjF')-col('min_RAdjF'))==0,0.0).otherwise((col('RAdjF')-col('min_RAdjF'))/(col('max_RAdjF')-col('min_RAdjF'))))\
    .select('user_id','listing_id','label','frequency','metadata_type','metadata_value','RAdjF','affinity')

    affinities.write.mode('overwrite').parquet(f'data/{ds}/affinities_parquet')
