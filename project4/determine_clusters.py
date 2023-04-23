import findspark
findspark.init()

import sys

import spacy
import unidecode
import contractions

import pyspark
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from matplotlib import pyplot as plt
from datetime import datetime
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.mllib.linalg import Vectors, VectorUDT


def draw_elbow_curve(clusters, costs):
    plt.plot(clusters, costs)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.savefig('elbow.png')


def plot_variance(pca_model):
    cum_values = pca_model.explainedVariance.cumsum() # get the cumulative values
    print('Cumulative PCA variance:', cum_values)
    # plot the graph 
    plt.figure(figsize=(10,8))
    plt.plot(range(1, cum_values.shape[0]), cum_values, marker = 'o', linestyle='--')
    plt.title('variance by components')
    plt.xlabel('num of components')
    plt.ylabel('cumulative explained variance')
    plt.savefig('variance.png')


def main():
    spark_conf = SparkConf()
    spark_conf.set('spark.dynamicAllocation.enabled', 'true')
    spark_conf.set('spark.driver.memory', '32g')

    # total executors memory is 128 gb, overhead ~10%, 
    # so total memory for executors is 128 - 128*0.1 = 115.2
    spark_conf.set('spark.executor.cores', '5')
    spark_conf.set('spark.executor.memoryOverhead', '13g')
    spark_conf.set('spark.executor.memory', '115g')

    spark = SparkSession.builder.master('local[*]').config(conf=spark_conf).appName('project4-clustering').getOrCreate()
    df = spark.read\
            .option('inferSchema','true')\
            .load(sys.argv[1])
    print('Number of partitions:', df.rdd.getNumPartitions())
    df.printSchema()
    print('Total number of records:', df.count())
    
    df.cache()
    ks = list(range(2, 20))
    costs = []
    models = []
    for k in ks:
        kmeans = KMeans(predictionCol=f'prediction_k{k}', featuresCol='features').setK(k).setSeed(444)
        model = kmeans.fit(df)
        inertia = model.summary.trainingCost
        print(f'{datetime.now()} - Fitted KMeans with k={k}. Inertia: {inertia}')
        costs.append(inertia)
        models.append(model)
        model.save(f'./kmeans-k{k}')
    draw_elbow_curve(ks, costs)
    for model in models:
        df = model.transform(df)
    df.select('_id', *[f'prediction_k{k}' for k in ks]).write.mode('Overwrite').save('./predictions')


if __name__ == '__main__':
    main()
