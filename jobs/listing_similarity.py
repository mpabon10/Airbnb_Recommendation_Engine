"""
listing_similarity.py
=====================
Pipeline Step 1: Listing Similarity Computation

Computes pairwise similarity scores between all Airbnb listings using a
weighted combination of five attributes:
  - Superhost status   (binary match,  weight = 0.20)
  - Neighbourhood      (binary match,  weight = 0.25)
  - Accommodates count (normalized diff, weight = 0.10)
  - Price              (normalized diff, weight = 0.35)
  - Review rating      (normalized diff, weight = 0.10)

For each listing, the top-100 most similar listings are retained and
written to Parquet for downstream use by simulate_txns.py.

Inputs:
  - data/nyc_listing_data_clean.csv          (full cleaned listing data)
  - data/nyc_listing_data_5pcnt_sample.csv   (5% sample used for pipeline)

Outputs:
  - data/top_100_sim_listings_parquet
"""

import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.window import Window
from builtins import min as minn
from builtins import max as maxx


spark = SparkSession.builder.master("local[4]").appName("Similarity").getOrCreate()

# ---------------------------------------------------------------------------
# 1. Load listing data with explicit schema
# ---------------------------------------------------------------------------
listing_schema = StructType([
    StructField("listing_id", IntegerType(), True),
    StructField("host_is_superhost", IntegerType(), True),
    StructField("neighbourhood_group_cleansed", StringType(), True),
    StructField("accommodates", IntegerType(), True),
    StructField("price", DoubleType(), True),
    StructField("review_scores_rating", DoubleType(), True)
])

# Read the full clean dataset (used only if you need to regenerate the 5% sample)
listing_df=spark.read.csv('data/nyc_listing_data_clean.csv',header=True, schema=listing_schema)

# NOTE: Uncomment the lines below to regenerate the 5% sample from the clean data.
# listing_df.sample(.05)\
#     .withColumn('listing_id',row_number().over(Window.orderBy('listing_id')))\
#     .write.mode('overwrite').csv('data/nyc_listing_data_5pcnt_sample.csv', header=True)

# Load the 5% sample that the rest of the pipeline operates on
listing_df=spark.read.csv('data/nyc_listing_data_5pcnt_sample.csv',header=True, schema=listing_schema)
listing_df.show()

# ---------------------------------------------------------------------------
# 2. Generate all pairwise listing combinations via cross join
# ---------------------------------------------------------------------------
base_df=listing_df
compare_df=listing_df

combos=base_df.crossJoin(compare_df)
combos.show()

# Rename columns with _x / _y suffixes to distinguish the two sides
base_columns = [f"{col}_x" for col in base_df.columns]
compare_columns = [f"{col}_y" for col in compare_df.columns]
combos = combos.toDF(*base_columns + compare_columns)
combos.show()

# Remove self-pairs (a listing compared to itself)
combos=combos.filter(col('listing_id_x')!=col('listing_id_y'))
combos.count()

# ---------------------------------------------------------------------------
# 3. Compute raw absolute differences for numeric attributes
# ---------------------------------------------------------------------------
combos_w_diff=(combos
    .withColumn('accommodates_diff', abs(col('accommodates_x')-col('accommodates_y')))
    .withColumn('price_diff', abs(col('price_x')-col('price_y')))
    .withColumn('rating_diff', abs(col('review_scores_rating_x')-col('review_scores_rating_y')))
    .withColumn('key',lit(1))
    )

# Compute the maximum observed difference for each numeric attribute.
# These maxima are used to min-max normalise the differences into [0, 1].
maxs=combos_w_diff.groupBy('key').agg(
    max('accommodates_diff').alias('max_accommodates_diff')
    , max('price_diff').alias('max_price_diff')
    , max('rating_diff').alias('max_rating_diff')
)

# ---------------------------------------------------------------------------
# 4. Compute weighted composite similarity score
#    Categorical attributes use exact-match (0 or 1).
#    Numeric attributes use 1 - (diff / max_diff) so higher = more similar.
# ---------------------------------------------------------------------------
superhost_wt,neighbourhood_wt,accommodates_wt,price_wt,rating_wt=.2,.25,.1,.35,.1

combos_w_sim=(combos_w_diff.join(maxs,'key','inner')
    .withColumn('superhost_sim', when(col('host_is_superhost_x')==col('host_is_superhost_y'),1).otherwise(0))
    .withColumn('neighbourhood_sim', when(col('neighbourhood_group_cleansed_x')==col('neighbourhood_group_cleansed_y'),1).otherwise(0))
    .withColumn('accommodates_sim', 1-(col('accommodates_diff')/col('max_accommodates_diff')))
    .withColumn('price_sim', 1-(col('price_diff')/col('max_price_diff')))
    .withColumn('rating_sim', 1-(col('rating_diff')/col('max_rating_diff')))
    .withColumn('listing_sim', 
        (superhost_wt*col('superhost_sim')
        + neighbourhood_wt*col('neighbourhood_sim')
        + accommodates_wt*col('accommodates_sim')
        + price_wt*col('price_sim')
        + rating_wt*col('rating_sim'))
        / (superhost_wt+neighbourhood_wt+accommodates_wt+price_wt+rating_wt)
        )
    .select('listing_id_x','listing_id_y'
        , 'listing_sim'
        , 'superhost_sim'
        , 'neighbourhood_sim'
        , 'accommodates_sim'
        , 'price_sim'
        , 'rating_sim'
    )
)

# ---------------------------------------------------------------------------
# 5. Rank similar listings and keep only the top 100 per listing
# ---------------------------------------------------------------------------
window_spec = Window.partitionBy("listing_id_x").orderBy(desc("listing_sim"),'listing_id_y')

combos_w_sim_rank=combos_w_sim\
    .withColumn("similarity_rank", dense_rank().over(window_spec))\
    .filter(col('similarity_rank')<=100)

combos_w_sim_rank.show()

# Write to a single coalesced Parquet file for downstream consumption
combos_w_sim_rank.coalesce(1).write.mode('overwrite').parquet('data/top_100_sim_listings_parquet')
