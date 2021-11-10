from py4j.java_gateway import JVMView
from pyspark import SparkContext
import pyspark
from pyspark.sql.types import StructType
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark import SparkConf
import json
sc = SparkContext("local[2]", "NetworkWordCount")
spark_session = pyspark.sql.SparkSession(sc)
ssc=StreamingContext(spark_session,1)
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
cols=["feature0","feature1"]
lines=ssc.socketTextStream('localhost',6100)
print('Hello!')
lines.map(lambda x: check_json(x, cols)).foreachRDD(lambda x: convert_json2df(x, cols))
ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate