import findspark
findspark.init()

import sys

import spacy
import unidecode
import contractions
import numpy as np

import time
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from pyspark.ml.feature import HashingTF, IDF, Word2Vec, Word2VecModel

import tokenize_articles


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


def cos_sim(a,b):
    r = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(r if not np.isnan(r) else -1e5) 

def main():
    spark_conf = SparkConf()
    spark_conf.set('spark.dynamicAllocation.enabled', 'true')
    spark_conf.set('spark.driver.memory', '16g')

    # total executors memory is 32 gb, overhead ~10%, 
    # so total memory for executors is 32 - 32*0.1 = 28.8
    spark_conf.set('spark.executor.cores', '5')
    spark_conf.set('spark.executor.memoryOverhead', '3g')
    spark_conf.set('spark.executor.memory', '16g')

    spark = SparkSession.builder.master('local[*]').config(conf=spark_conf).appName('project4-clustering').getOrCreate()
    
    tokenized_path, paper_title, N = sys.argv[1:]
    df = spark.read\
            .option('inferSchema','true')\
            .load(tokenized_path)
    df.printSchema()
    print('Total number of records:', df.count())
    print('Paper title:', paper_title)

    tokenizer = PapersTokenizer()
    tokens = tokenizer(paper_title)
    print('Extracted tokens:', tokens)

    w2v_model = Word2VecModel.load('./word2vec')
    search = spark.createDataFrame([(tokens,)], ['tokens'])

    search = w2v_model.transform(search)
    search.show()

    cosim = F.udf(cos_sim, FloatType())
    
    # df = df.withColumn("coSim", cosim(F.col("features"), array([lit(v) for v in static_array])))
    df = df.withColumn('cosim', cosim(F.col('features'), F.array([F.lit(x) for x in search.first().features])))
    print(f'Papers with the top {N} similarities:')
    df = df.sort(F.col('cosim').desc()).select('_id', 'title', 'abstract', 'cosim').limit(int(N))
    df.show(int(N))
    save_fp = f'./search{time.time()}'
    df.coalesce(1).write.csv(save_fp)
    print('Saved the result to', save_fp)


if __name__ == '__main__':
    main()
