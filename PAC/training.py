import numpy as np 
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.sql.types import Row, StringType, StructField, StructType
from pyspark.streaming import StreamingContext
import json
import pickle
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer,StopWordsRemover, CountVectorizer,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import count, length
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, split
from pyspark.ml.feature import CountVectorizer,StopWordsRemover,StringIndexer
from pyspark.ml.classification import NaiveBayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
sc = SparkContext("local[2]", "spam").getOrCreate() #no of threads to run it, cluster name 
ssc = StreamingContext(sc, 1)
spark = SparkSession(sc)
lines = ssc.socketTextStream("localhost", 6100)
# Use defaults
nb = NaiveBayes()
count_batch_number=0
def readStream(rdd):
  
  global count_batch_number
  count_batch_number+=1
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
    #data = df.withColumn('length',length(df['content']))
    #tokenizer = Tokenizer(inputCol="content", outputCol="token_content")
    #stopremove = StopWordsRemover(inputCol='token_content',outputCol='stop_tokens')
    #count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='count_vec')
    #idf = IDF(inputCol="count_vec", outputCol="tf_idf")
    #ham_spam_to_num = StringIndexer(inputCol='verdict',outputCol='label')
    #clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')
    #data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
    #cleaner = data_prep_pipe.fit(data)
    #clean_data = cleaner.transform(data)
    data = df.withColumn('length',length(df['content']))
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
    #clean_data.show()
    clean_data_np_X=numpy.array(clean_data.select('features').collect())
    #print(clean_data_np_X)
    clean_data_np_y=numpy.array(clean_data.select('numericlabel').collect()).flatten()
    clean_data_np_X=[i.flatten() for i in clean_data_np_X]
    clean_data_np_X=numpy.array(clean_data_np_X)
    #nb=MultinomialNB()
    #print(clean_data_np_X)
    #print(clean_data_np_y)
    #clf = make_pipeline(StandardScaler(),
                     #SGDClassifier(max_iter=1000, tol=1e-3))
    #spam_predictor = nb.partial_fit(clean_data_np_X,clean_data_np_y,classes=numpy.unique(clean_data_np_y))
    pac=PassiveAggressiveClassifier(max_iter=1000, random_state=5,tol=1e-3,C=0.5,max_iter=50)
    pac_model=pac.partial_fit(clean_data_np_X, clean_data_np_y,classes=numpy.unique(clean_data_np_y))
    print(count_batch_number)
    if count_batch_number==60:
        filename='pac_model'
        with open(filename,'wb') as f:
            pickle.dump(pac_model,f)
            print('dumped!')

lines.foreachRDD( lambda rdd: readStream(rdd) )
results = []
ssc.start()
ssc.awaitTermination()
