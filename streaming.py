from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.sql.types import IntegerType, Row, StringType, StructField, StructType
from pyspark.streaming import StreamingContext
import json
sc = SparkContext("local[2]", "sentiment").getOrCreate() #no of threads to run it, cluster name 
ssc = StreamingContext(sc, 1)
spark = SparkSession(sc)


#streamingContext.textFileStream(/home/PES1UG19CS493/Downloads/Project/sentiment)#file streaming 
lines = ssc.socketTextStream("localhost", 6100)
# words = lines.flatMap(lambda line: line.split(" "))
# pairs = words.map(lambda word: (word, 1))
# wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Print the first ten elements of each RDD generated in this DStream to the console
#wordCounts.collect()
#wordCounts.pprint()
def readStream(rdd):
  if not rdd.isEmpty():
    #df = spark.read.json(rdd)
    #df = df.select(F.array(F.expr("0.*")).alias("something"))
    #df.printSchema()
    rddStream=rdd.collect()
    array_of_vals=[i for rdd_val in rddStream for i in list(json.loads(rdd_val).values())]
    # print('Started the Process')
    # print('Selection of Columns')
    # df = df.select(F.expr('0.feature0').alias('Sentiment'),F.expr('0.feature1').alias('Tweet'))
    # print('whats happening')
    # print(df.show())
    #print(values)
    schema=StructType([
      StructField('subject',StringType(),False),
      StructField('content',StringType(),False),
      StructField('verdict',StringType(),False)
    ])
    df=spark.createDataFrame((Row(**d) for d in array_of_vals),schema)
    df=df['content','verdict']
    df.show()
lines.foreachRDD( lambda rdd: readStream(rdd) )
ssc.start()
ssc.awaitTermination()
