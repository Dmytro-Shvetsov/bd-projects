import findspark
findspark.init()

import sys

import spacy
import unidecode
import contractions

import pyspark
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import HashingTF, IDF, Word2Vec


class PapersTokenizer:
    additional_stopwords = {
        'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 'rights',
        'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 'al.', 'Elsevier',
        'PMC', 'CZI', 'www'
    }

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.nlp.Defaults.stop_words.update(self.additional_stopwords)

    @staticmethod
    def remove_accented_chars(text):
        """remove accented characters from text, e.g. café"""
        text = unidecode.unidecode(text)
        return text

    @staticmethod
    def expand_contractions(text):
        """expand shortened words, e.g. don't to do not"""
        text = contractions.fix(text)
        return text
    
    def __call__(self, text, verbose=False):
        try:
            text = self.remove_accented_chars(text)
            text = self.expand_contractions(text)
            if verbose: 
                print('Text:', text)
            lemmas = list(
                t.lemma_.lower() 
                for t in self.nlp(text) if
                    t.is_alpha and \
                    not t.is_punct and \
                    not t.is_space and \
                    not t.is_stop and len(t)>1
            )
            return lemmas
        except Exception as exc:
            return [str(exc)]


def main():
    spark_conf = SparkConf()
    spark_conf.set('spark.sql.files.maxPartitionBytes', 600 * 1048576)
    spark_conf.set('spark.dynamicAllocation.enabled', 'true')
    spark_conf.set('spark.driver.memory', '32g')

    # total executors memory is 128 gb, overhead ~10%, 
    # so total memory for executors is 128 - 128*0.1 = 115.2
    spark_conf.set('spark.executor.cores', '5')
    spark_conf.set('spark.executor.memoryOverhead', '13g')
    spark_conf.set('spark.executor.memory', '115g')

    # spark_conf.set('spark.sql.adaptive.coalescePartitions.initialPartitionNum', 21)
    # spark_conf.set('spark.sql.adaptive.coalescePartitions.parallelismFirst', 'false')

    spark = SparkSession.builder.master('local[*]').config(conf=spark_conf).appName('project4-clustering').getOrCreate()
    # "../datasets/chunks/*.json"
    df = spark.read\
            .option('inferSchema','true')\
            .json(sys.argv[1] + '/*.json')
    print('Number of partitions:', df.rdd.getNumPartitions())
    df.printSchema()
    print('Total number of records:', df.count())
    df_en = df.filter(df.lang == 'en').select('_id', 'title', 'abstract')
    print('Total number of english records:', df_en.count())

    df_en = df_en.na.fill('')
    df_en = df_en.withColumn('text', F.concat_ws(' ', df_en.title, df_en.abstract))
    df_en.show(3)

    df_en = df_en.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    tokenizer = PapersTokenizer()

    tokenize = spark.udf.register('tokenize', tokenizer, returnType=ArrayType(StringType()))
    df_en_tokenized = df_en.withColumn('tokens', tokenize(df_en.text))
    df_en_tokenized.show(5)

    df_en_tokenized = df_en_tokenized.filter(F.size(df_en_tokenized.tokens) > 0)

    df_en_tokenized = df_en_tokenized.persist(pyspark.StorageLevel.DISK_ONLY)

    word2vec = Word2Vec(vectorSize=100, seed=444, inputCol='tokens', outputCol='features', maxIter=10)
    model = word2vec.fit(df_en_tokenized)
    model.save('./word2vec')
    
    rescaled_data = model.transform(df_en_tokenized)
    rescaled_data.write.mode('Overwrite').save('./w2v_tokenized')


if __name__ == '__main__':
    main()
