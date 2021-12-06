import numpy
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.sql.types import Row, StringType, StructField, StructType
from pyspark.streaming import StreamingContext
import json
import pickle
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import length
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import CountVectorizer ,StopWordsRemover,StringIndexer
from sklearn.cluster import MiniBatchKMeans,KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
sc = SparkContext("local[2]", "sentiment").getOrCreate() #no of threads to run it, cluster name 
ssc = StreamingContext(sc, 1)
spark = SparkSession(sc)
lines = ssc.socketTextStream("localhost", 6100)
count_batch_number=0
def readStream(rdd):
  global count_batch_number
  count_batch_number+=1
  print(count_batch_number)
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
    stages=[]
    regexTokenizer=RegexTokenizer(inputCol='content',outputCol='token',pattern='\\W+')
    stages+=[regexTokenizer]
    stopremove = StopWordsRemover(inputCol='token',outputCol='stop_tokens')
    stages+=[stopremove]
    cv=CountVectorizer(inputCol='stop_tokens',outputCol='token_features',minDF=2.0,vocabSize=2000)
    stages+=[cv]
    indexer=StringIndexer(inputCol='verdict',outputCol='numericlabel')
    stages+=[indexer]
    vectorassemble=VectorAssembler(inputCols=['token_features','length'],outputCol="features")
    stages+=[vectorassemble]
    pipeline=Pipeline(stages=stages)
    clean_data=pipeline.fit(data).transform(data)
    #clean_data.show()
    clean_data_np=numpy.array(clean_data.select('features').collect())
    clean_data_np=[i.flatten() for i in clean_data_np]
    clean_data_np=numpy.array(clean_data_np)
    #clean_data.show()
    #(training,validate) = clean_data_np.randomSplit([0.7,0.3])
    #nsamples, nx, ny = clean_data_np.shape
    #clean_data_np = clean_data_np.reshape((nsamples,nx*ny))
    kmeans=MiniBatchKMeans(n_clusters=2,batch_size=500,init='k-means++')
    kmeans_model = kmeans.partial_fit(clean_data_np)
    if count_batch_number==60:
        filename='clustering_model'
        with open(filename,'wb') as f:
            pickle.dump(kmeans_model,f)
            print('dumped!')
        print(kmeans_model.cluster_centers_)
        print(kmeans.labels_)
        k_means = KMeans(init="k-means++", n_clusters=2, n_init=10)
        k_means.fit(clean_data_np)
        k_means_labels = k_means.labels_
        k_means_cluster_centers = k_means.cluster_centers_
        k_means_labels_unique = numpy.unique(k_means_labels)
# with minibatchkmeans
        mbk_means_labels = kmeans_model.labels_
        mbk_means_cluster_centers = kmeans_model.cluster_centers_
        #mbk_means_labels_unique = numpy.unique(mbk_means_labels)
# Plot result
        fig = plt.figure(figsize=(8, 3))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
        colors = ['#4EACC5', '#FF9C34']

# # We want to have the same colors for the same cluster from the
# # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# # closest one.
        distance = euclidean_distances(k_means_cluster_centers,
                               mbk_means_cluster_centers,
                                squared=True)
        order = distance.argmin(axis=1)
  # KMeans
        ax = fig.add_subplot(1, 2,1)
        for k, col in zip(range(2), colors):
            my_members = k_means_labels == k
            cluster_center = k_means_cluster_centers[k]
            ax.plot(clean_data_np[my_members, 0], clean_data_np[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                   markeredgecolor='k', markersize=6)
        ax.set_title('KMeans')
        ax.set_xticks(())
        ax.set_yticks(())
# # MiniBatchKMeans
        ax = fig.add_subplot(1, 2, 2)
        for k, col in zip(range(2), colors):
            my_members = mbk_means_labels == order[k]
            cluster_center = mbk_means_cluster_centers[order[k]]
            ax.plot(clean_data_np[my_members, 0], clean_data_np[my_members, 1], 'w',
              markerfacecolor=col, marker='.')
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                  markeredgecolor='k', markersize=6)
        ax.set_title('MiniBatchKMeans')
        ax.set_xticks(())
        ax.set_yticks(())
        plt.show()

lines.foreachRDD( lambda rdd: readStream(rdd) )

ssc.start()
ssc.awaitTermination()