import numpy
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.sql.types import Row, StringType, StructField, StructType
from pyspark.streaming import StreamingContext
import json
import pickle
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, RegexTokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import length
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import CountVectorizer,StopWordsRemover,StringIndexer
from pyspark.ml.classification import NaiveBayes
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
sc = SparkContext("local[2]", "sentiment").getOrCreate() #no of threads to run it, cluster name 
ssc = StreamingContext(sc, 1)
spark = SparkSession(sc)
lines = ssc.socketTextStream("localhost", 6100)

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
    # tokenizer = Tokenizer(inputCol="content", outputCol="token_content")
    # stopremove = StopWordsRemover(inputCol='token_content',outputCol='stop_tokens')
    # count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='count_vec')
    # idf = IDF(inputCol="count_vec", outputCol="tf_idf")
    # ham_spam_to_num = StringIndexer(inputCol='verdict',outputCol='label')
    # clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')
    # data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
    # cleaner = data_prep_pipe.fit(data)
    # clean_data = cleaner.transform(data)
    stages=[]
    regexTokenizer=RegexTokenizer(inputCol='content',outputCol='token',pattern='\\W+')
    stages+=[regexTokenizer]
    stopremove = StopWordsRemover(inputCol='token',outputCol='stop_tokens')
    stages+=[stopremove]
    hv=CountVectorizer(inputCol='stop_tokens',outputCol='token_features',minDF=2.0,vocabSize=700)
    stages+=[hv]
    #hashing_stage = HashingTF(inputCol="stop_tokens", outputCol="hashed_features")
    #stages+=[hashing_stage]
    #idf_stage = IDF(inputCol="hashed_features", outputCol="features", minDocFreq=1)
    #stages+=[idf_stage]
    indexer=StringIndexer(inputCol='verdict',outputCol='numericlabel')
    stages+=[indexer]
    vectorassemble=VectorAssembler(inputCols=['token_features','length'],outputCol="features")
    stages+=[vectorassemble]
    pipeline=Pipeline(stages=stages)
    clean_data=pipeline.fit(data).transform(data)
    clean_data_np_X=numpy.array(clean_data.select('features').collect())
    #print(clean_data_np_X)
    clean_data_np_X=[i.flatten() for i in clean_data_np_X]
    clean_data_np_X=numpy.array(clean_data_np_X)
    #clean_data_np_X=numpy.pad(clean_data_np_X,2741-len(clean_data_np_X),'edge')
    #print(type(clean_data_np_X))
    filename='nb_model'
    load_nbmodel=pickle.load(open(filename,'rb'))
    #test_results = load_nbmodel.predict(clean_data_np_X)
    actual_test_op=numpy.array(clean_data.select('numericlabel').collect()).flatten()
    #accuracy=load_nbmodel.score(clean_data_np_X,actual_test_op)
    test_output=load_nbmodel.predict(clean_data_np_X)

    #metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
    #print(metrics.confusionMatrix().toArray())
    print(accuracy_score(actual_test_op,test_output)*100,precision_score(actual_test_op,test_output),recall_score(actual_test_op,test_output),f1_score(actual_test_op,test_output))

lines.foreachRDD( lambda rdd: readStream(rdd) )
ssc.start()
ssc.awaitTermination()