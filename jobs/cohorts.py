"""
cohorts.py
==========
Pipeline Step 5: RFM Cohort Assignment via K-Means Clustering

Segments users into cohorts based on their Recency, Frequency, and Monetary
(RFM) values using K-Means clustering (k=4).

Workflow:
  1. Load the labelled RFM data from ones_n_zeros.py.
  2. Cap extreme values at the observed maximum to reduce outlier impact.
  3. Min-max scale each RFM dimension to [0, 1].
  4. Apply configurable weights (R=0.3, F=0.4, M=0.3) to emphasise frequency.
  5. Run K-Means with 4 clusters and assign an ordinal RFM_Cohort label
     ranked by descending average frequency.

The resulting cohort labels are used by affinities.py to normalise user
affinity scores relative to their peer group.

Inputs:
  - data/{training,scoring}/ones_and_zeros_rfm_parquet  (from ones_n_zeros.py)

Outputs:
  - data/{training,scoring}/cohorts_parquet
"""

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder.master("local[4]").appName("Cohorts").getOrCreate()

data_sets=['training','scoring']
for ds in data_sets:
    # -----------------------------------------------------------------
    # 1. Load RFM data and compute upper-bound / min values for scaling
    # -----------------------------------------------------------------
    num_RFM_cohorts=4
    
    ones_and_zeros_rfm=spark.read.parquet(f'data/{ds}/ones_and_zeros_rfm_parquet')\
        .withColumn('key',lit(1))

    # RFM dimension weights – frequency is most important for cohort segmentation
    weight_freq = 0.4
    weight_recency = 0.3 
    weight_monetary = 0.3

    # Compute upper-bound and minimum values for min-max scaling
    upper_bounds=(ones_and_zeros_rfm.groupBy('key').agg(
        # percentile_approx('recency', 0.98).alias('upper_bound_r')
        # , percentile_approx('frequency', 0.98).alias('upper_bound_f')
        # , percentile_approx('monetary', 0.98).alias('upper_bound_m')
        max('recency').alias('upper_bound_r')
        , max('frequency').alias('upper_bound_f')
        , max('monetary').alias('upper_bound_m')
        , min('recency').alias('min_recency')
        , min('frequency').alias('min_frequency')
        , min('monetary').alias('min_monetary')
        )
    )
    upper_bounds.persist()

    # -----------------------------------------------------------------
    # 2. Cap, scale, and weight the RFM dimensions
    # -----------------------------------------------------------------
    ones_and_zeros_rfm_capped_scaled_weighted=(ones_and_zeros_rfm
    .join(upper_bounds, 'key', 'inner')
    # capped upper bound
    .withColumn('recency_cap', when(col('recency')>col('upper_bound_r'), col('upper_bound_r')).otherwise(col('recency')))
    .withColumn('frequency_cap', when(col('frequency')>col('upper_bound_f'), col('upper_bound_f')).otherwise(col('frequency')))
    .withColumn('monetary_cap', when(col('monetary')>col('upper_bound_m'), col('upper_bound_m')).otherwise(col('monetary')))
    #scaled
    .withColumn('recency_scaled',(col('recency_cap')-col('min_recency'))/(col('upper_bound_r')-col('min_recency'))) #removed min subtractin since hard coded to 0
    .withColumn('frequency_scaled',(col('frequency_cap')-col('min_frequency'))/(col('upper_bound_f')-col('min_frequency')))
    .withColumn('monetary_scaled',(col('monetary_cap')-col('min_monetary'))/(col('upper_bound_m')-col('min_monetary'))) #removed min subtraction since hard coded to 0
    # weighted 
    .withColumn('recency_weighted', col('recency_scaled') * weight_recency)
    .withColumn('frequency_weighted', col('frequency_scaled') * weight_freq)
    .withColumn('monetary_weighted', col('monetary_scaled') * weight_monetary)
    .select('user_id','listing_id','label','recency','frequency','monetary','recency_weighted','frequency_weighted','monetary_weighted')
    )

    # -----------------------------------------------------------------
    # 3. Run K-Means clustering on the weighted RFM features
    # -----------------------------------------------------------------
    assembler = VectorAssembler(inputCols=['recency_weighted', 'frequency_weighted', 'monetary_weighted'], outputCol='features')
    assembled_df = assembler.transform(ones_and_zeros_rfm_capped_scaled_weighted)

    kmeans=KMeans(k=num_RFM_cohorts, seed=2195, featuresCol='features', predictionCol='cluster')

    modeling_data= assembled_df

    model = kmeans.fit(modeling_data)
    # Make predictions (assign cluster to each row)
    predictions = model.transform(assembled_df)

    # -----------------------------------------------------------------
    # 4. Map numeric cluster IDs to ordinal cohort labels (1 = highest
    #    frequency, 4 = lowest) and write results
    # -----------------------------------------------------------------
    cohort_mapping=(predictions
    .groupBy('cluster')
    .agg(mean('frequency').alias('avg_frequency')
    , min('frequency').alias('min_frequency')
    , max('frequency').alias('max_frequency')
    ,count('*').alias('num_users')
    )
    .withColumn('RFM_Cohort', row_number().over(Window.orderBy(desc('avg_frequency'))))
    )

    ones_and_zeros_rfm_cohort=(predictions
    .join(cohort_mapping.select('cluster','RFM_Cohort'), 'cluster', 'inner')
    .select('user_id','listing_id','label','recency','frequency','monetary','RFM_Cohort')
    )

    # Write cohort-enriched data for downstream use by affinities.py
    ones_and_zeros_rfm_cohort.write.mode('overwrite').parquet(f'data/{ds}/cohorts_parquet')
    upper_bounds.unpersist()

