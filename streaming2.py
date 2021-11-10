from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
import json

from pyspark.sql.types import StructType
spark = SparkSession \
    .builder \
    .appName("StructuredNetworkWordCount") \
    .getOrCreate()
def check_json(js, col):
    try:
        data = json.loads(js)
        return [data.get(i) for i in col]
    except:
        return []
def convert_json2df(rdd, col):
    ss = SparkSession(rdd.context)
    if rdd.isEmpty():
        return
    df = ss.createDataFrame(rdd, schema=StructType("based on 'col'"))
    print(df.show(10))
lines = spark \
    .readStream \
    .format("socket") \
    .option("host", "localhost") \
    .option("port", 6100) \
    .load()
cols=["feature0","feature1"]
# Split the lines into words
# words = lines.select(
#    explode(
#        split(lines.value, " ")
#    ).alias("word")
# )

# Generate running word count
# wordCounts = words.groupBy("word").count()
query = lines \
     .writeStream \
     .outputMode("complete") \
     .format("console") \
     .start()

query.awaitTermination()