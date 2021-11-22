
from pyspark.sql import SparkSession
import json
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import col
sc = SparkContext("local[2]", "sentiment") #no of threads to run it, cluster name 
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)

#streamingContext.textFileStream(/home/PES1UG19CS493/Downloads/Project/sentiment)#file streaming 
lines = ssc.socketTextStream("localhost", 6100)
lines.pprint()
#words = lines.flatMap(lambda line: spark_parallelize(line))
#words.pprint()
#print(words)
#words = lines.flatMap(lambda line: json.loads(line))
#pairs = words.map(lambda word: (word, 1))
#wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Print the first ten elements of each RDD generated in this DStream to the console
#wordCounts.collect()
#wordCounts.pprint()
def readMyStream(rdd):
    if not rdd.isEmpty():
        df = spark.read.json(rdd)
        print('Started the Process')
        print('Selection of Columns')
        df = df.select('feature0','feature1')
        df.show()
def spark_parallelize(line):
    inter=sc.parallelize(line)
    return spark.read.json(inter)
lines.foreachRDD( lambda rdd: readMyStream(rdd) )

ssc.start()
ssc.awaitTermination()
