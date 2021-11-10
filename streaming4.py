# importing required libraries
from pyspark import SparkContext
import pyspark
from pyspark.streaming import StreamingContext
import pyspark.sql.types as tp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import Row

# initializing spark session
sc = SparkContext("local[2]",appName="PySparkShell")
spark = pyspark.sql.SparkSession(sc)
    
# define the schema
my_schema = tp.StructType([
  tp.StructField(name= 'feature0', dataType= tp.IntegerType(),  nullable= True),
  tp.StructField(name= 'feature1', dataType= tp.StringType(),   nullable= True)
])
    
ssc = StreamingContext(spark, batchDuration= 3)
lines = ssc.socketTextStream('localhost', 6100)
words = lines.flatMap(lambda line : line.split(','))
# read the dataset  
words.foreachRDD(lambda x:print(x.show()))
