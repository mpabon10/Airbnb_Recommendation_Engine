import pandas as pd
import numpy as np
import shutil
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import * 
from pyspark.sql.types import StructType, StructField, IntegerType, BooleanType, StringType, DoubleType
from builtins import min as minn
from builtins import max as maxx


spark = SparkSession.builder.master("local[4]").appName("TestApp").getOrCreate()


# Define the schema
listing_schema = StructType([
    StructField("listing_id", IntegerType(), True),
    StructField("host_is_superhost", IntegerType(), True),
    StructField("neighbourhood_group_cleansed", StringType(), True),
    StructField("accommodates", IntegerType(), True),
    StructField("price", DoubleType(), True),
    StructField("review_scores_rating", DoubleType(), True)
])

listing_df=spark.read.csv('data/nyc_listing_data_5pcnt_sample.csv',header=True, schema=listing_schema)
listing_df.show()

max_listing_id = listing_df.select(max("listing_id")).collect()[0][0]

# Function to generate an exponentially distributed random number between 1 and max_num
def exponential_random(max_num):
    # Scale factor (λ) controls the steepness of the exponential decay
    # Higher lambda means steeper decay (fewer higher values)
    lambda_value = 10.5  # you can adjust this for a different decay rate
    
    # Generate an exponentially distributed number
    exp_random = np.random.exponential(1 / lambda_value)
    
    # Scale it to the range [1, max_num], using max_num as the upper bound
    scaled_value = minn(max_num, int(np.ceil(exp_random * max_num)))
    
    # Ensure the value is at least 1
    return maxx(1, scaled_value)

# Register the function as a UDF in Spark
exp_random_udf = udf(exponential_random, IntegerType())

# Set the max frequency
num_users=10000
max_freq = 30
#user IDs
user_df=spark.range(1, num_users+1)\
    .withColumnRenamed('id','user_id')\
    .withColumn("frequency", exp_random_udf(expr(str(max_freq))))

user_df.write.mode('overwrite').parquet('data/users_parquet')

user_df=spark.read.parquet('data/users_parquet')

sim_listings=spark.read.parquet('data/top_100_sim_listings_parquet')\
    .select(col('listing_id_x').alias('listing_id'),col('listing_id_y').alias('sim_listing_id'),'similarity_rank')

# sim_listings.filter(col('listing_id')==671).show()

max_sim_listing=sim_listings.select(max("similarity_rank")).collect()[0][0]


txns_path='data/transactions_parquet'
# Check if the directory exists
if os.path.exists(txns_path):
    # Remove the directory and its contents
    shutil.rmtree(txns_path)
    print(f"Directory {txns_path} has been deleted.")
else:
    print(f"Directory {txns_path} does not exist.")

for booking in range(1,max_freq+1):
    print(booking)
    subset=user_df.filter(col('frequency') >= lit(booking))\
        .select('user_id')
    
    if subset.count()>0:

        if booking==1:
            transactions=subset.withColumn('rand_val', rand(seed=2195))\
                .withColumn('listing_id',floor(col('rand_val') * (max_listing_id - 1) + 1).cast('int'))\
                .join(listing_df.select('listing_id','price'),'listing_id','left')\
                .withColumn('booking_num',lit(booking))\
                .withColumn('days_since_last',lit(0))\
                .withColumn('running_avg_price',col('price'))\
                .drop('rand_val')
            
            transactions.write.mode('overwrite').parquet('data/transactions_parquet/booking_num='+str(booking)+'/')
        else:
            hist_transactions=spark.read.parquet('data/transactions_parquet/booking_num='+str(booking-1)+'/')
            
            random_next_listing=hist_transactions.select('user_id','listing_id','running_avg_price',col('price').alias('last_price'))\
                .join(subset,'user_id','inner')\
                .withColumn('similarity_rank',exp_random_udf(expr(str(max_sim_listing))))\
            
            random_next_listing.persist()
                
            transactions=random_next_listing\
                .join(sim_listings,['listing_id','similarity_rank'],'left')\
                .drop('listing_id')\
                .withColumnRenamed('sim_listing_id','listing_id')\
                .join(listing_df.select('listing_id','price'),'listing_id','left')\
                .withColumn('booking_num',lit(booking))\
                .withColumn('days_since_last',(rand() * 100 + 1).cast("int"))\
                .withColumn('running_avg_price',((col('running_avg_price')*(col('booking_num')-1))+col('last_price'))/(col('booking_num')))


            
            transactions.write.mode('overwrite').parquet('data/transactions_parquet/booking_num='+str(booking)+'/')
            random_next_listing.unpersist()
    else:
        print(f'no users with {booking} bookings')



transactions=spark.read.parquet('data/transactions_parquet')
transactions.count()
transactions.orderBy('user_id','booking_num').show()
# user_df.groupBy(lit(1)).agg(sum('frequency')).show()
# user_df.filter(col('user_id')==54).show()
user_txns=transactions.groupBy('user_id').agg(count('*').alias('tot_txns'))
user_df.join(user_txns, 'user_id','left').filter(col('frequency')!=col('tot_txns')).show()

# transactions.filter(col('user_id')==54).orderBy('booking_num').show()
# sim_listings.filter(col('listing_id_x')==272).show()

user_df.filter(col('user_id')==20).show()
transactions.filter(col('user_id')==20).show()