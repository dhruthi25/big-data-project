import numpy
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.sql.types import Row, StringType, StructField, StructType
from pyspark.streaming import StreamingContext
import json
import pickle
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import length
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, split
from pyspark.ml.feature import CountVectorizer,StopWordsRemover,StringIndexer
from pyspark.ml.classification import NaiveBayes

sc = SparkContext("local[2]", "sentiment").getOrCreate() #no of threads to run it, cluster name 
ssc = StreamingContext(sc, 1)
spark = SparkSession(sc)
lines = ssc.socketTextStream("localhost", 6100)
# Use defaults
nb = NaiveBayes()

def readStream(rdd):
  if not rdd.isEmpty():
    rddStream=rdd.collect()
    array_of_vals=[i for rdd_val in rddStream for i in list(json.loads(rdd_val).values())]
    schema=StructType([
      StructField('subject',StringType(),False),
      StructField('content',StringType(),False),
      StructField('verdict',StringType(),False)
    ])
    df=spark.createDataFrame((Row(**d) for d in array_of_vals),schema)
    df=df['content','verdict']
    #df.show()
    data = df.withColumn('length',length(df['content']))
    tokenizer = Tokenizer(inputCol="content", outputCol="token_content")
    stopremove = StopWordsRemover(inputCol='token_content',outputCol='stop_tokens')
    count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='count_vec')
    idf = IDF(inputCol="count_vec", outputCol="tf_idf")
    ham_spam_to_num = StringIndexer(inputCol='verdict',outputCol='label')
    clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')
    data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
    cleaner = data_prep_pipe.fit(data)
    clean_data = cleaner.transform(data)
    clean_data_np_X=numpy.array(clean_data.select('features').collect())
    #print(clean_data_np_X)
    clean_data_np_X=[i.flatten() for i in clean_data_np_X]
    clean_data_np_X=numpy.array(clean_data_np_X)
    #clean_data_np_X=numpy.pad(clean_data_np_X,2741-len(clean_data_np_X),'edge')
    print(type(clean_data_np_X))
    filename='nb_model'
    load_nbmodel=pickle.load(open(filename,'rb'))
    #test_results = load_nbmodel.predict(clean_data_np_X)
    clean_data_np_y=numpy.array(clean_data.select('label').collect()).flatten()
    accuracy=load_nbmodel.score(clean_data_np_X,clean_data_np_y)
    print("Accuracy:{}".format(accuracy))
    #metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
    #print(metrics.confusionMatrix().toArray())
    #print("Accuracy:{}".format(metrics.accuracy))
    #print("Precision:{}".format(metrics.precision(1.0)))
    #print("Recall:{}".format(metrics.recall(1.0)))
    #print("F1 Score:{}".format(metrics.fMeasure(1.0)))

lines.foreachRDD( lambda rdd: readStream(rdd) )
ssc.start()
ssc.awaitTermination()