from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.sql.types import IntegerType, Row, StringType, StructField, StructType
from pyspark.streaming import StreamingContext
import json
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.sql.functions import length
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import re
import pyspark
from pyspark.sql import SparkSession,SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import col, split
from pyspark.ml.feature import RegexTokenizer, CountVectorizer, PCA ,StopWordsRemover,StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier,LogisticRegression,NaiveBayes,LinearSVC
from pyspark.ml.clustering import KMeans
from pyspark.sql.types import DoubleType

sc = SparkContext("local[2]", "sentiment").getOrCreate() #no of threads to run it, cluster name 
ssc = StreamingContext(sc, 1)
spark = SparkSession(sc)
lines = ssc.socketTextStream("localhost", 6100)      
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
    #clean_data.show()
    clean_data = clean_data.select(['label','features'])
    #clean_data.show()
    (training,validate) = clean_data.randomSplit([0.7,0.3])
    lsvc = LinearSVC(maxIter=10, regParam=0.1)
    # Fit the model
    lsvcModel = lsvc.fit(training)
    predictions = lsvcModel.transform(validate)
    #kmeans = KMeans().setK(2).setSeed(1)
    #kmodel = kmeans.fit(training)
    #predictions=kmodel.transform(validate)
    # converting column datatype from integer to double
    predictions = predictions.withColumn("prediction", predictions["prediction"].cast(DoubleType()))
    predictions.show()
    #acc_eval = MulticlassClassificationEvaluator()
    #acc = acc_eval.evaluate(test_results)
    preds_and_labels=predictions.select(['prediction','label'])
    #print("Accuracy of model at predicting spam was: {}".format(acc))
    metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
    print(metrics.confusionMatrix().toArray())
    print("Accuracy:{}".format(metrics.accuracy))
    print("Precision:{}".format(metrics.precision(1.0)))
    print("Recall:{}".format(metrics.recall(1.0)))
    print("F1 Score:{}".format(metrics.fMeasure(1.0)))

lines.foreachRDD( lambda rdd: readStream(rdd) )
ssc.start()
ssc.awaitTermination()
