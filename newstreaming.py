

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
spark = SparkSession.builder().getOrCreate()
sc = SparkContext("local[2]", "sentiment") #no of threads to run it, cluster name 
ssc = StreamingContext(sc, 1)

socketDF = spark \
    .readStream()  \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 6100) \
    .load()

socketDF.isStreaming()    # Returns True for DataFrames that have streaming sources

socketDF.printSchema() 
ssc.start()
ssc.awaitTermination()