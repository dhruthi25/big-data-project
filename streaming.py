
from pyspark.sql import SparkSession

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext("local[2]", "sentiment") #no of threads to run it, cluster name 
ssc = StreamingContext(sc, 1)



#streamingContext.textFileStream(/home/PES1UG19CS493/Downloads/Project/sentiment)#file streaming 
lines = ssc.socketTextStream("localhost", 6100)
words = lines.flatMap(lambda line: line.split(" "))
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Print the first ten elements of each RDD generated in this DStream to the console
#wordCounts.collect()
wordCounts.pprint()


ssc.start()
ssc.awaitTermination()
