# install spark https://spark.apache.org/downloads.html
#
#    pip install pyspark
# 
# pour windows https://medium.com/big-data-engineering/how-to-install-apache-spark-2-x-in-your-pc-e2047246ffc3
#
# run with python like
#
#    python -i iris-spark.py
#
# "-i" option is for interactive mode
# 
# It should allow you to experiment in the python REPL
# without need to re-run all this every time


import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType
from pyspark.mllib.linalg.distributed import RowMatrix

# lancement de spark
spark = SparkSession\
        .builder\
        .appName("RedWine_App")\
        .getOrCreate()
sqlContext = SQLContext(sparkContext = spark.sparkContext, sparkSession = spark)

# chargement des données du dataSet
nullable = True
schema = StructType([
    StructField("fixed_acidity", FloatType(), nullable),
    StructField("volatile_acidity", FloatType(), nullable),
    StructField("citric_acid", FloatType(), nullable),
    StructField("residual_sugar", FloatType(), nullable),
    StructField("chlorides", FloatType(), nullable),
    StructField("free_sulfur_dioxide", FloatType(), nullable),
    StructField("total_sulfur_dioxide", FloatType(), nullable),
    StructField("density", FloatType(), nullable),
    StructField("ph", FloatType(), nullable),
    StructField("sulphates", FloatType(), nullable),
    StructField("alcohol", FloatType(), nullable),
    StructField("quality", IntegerType(), nullable),
])
RedWineQuality = sqlContext.read.csv('winequality-red.csv', header = False, schema = schema)

# répartition en colonne
vecAssembler = VectorAssembler(
    inputCols=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide','density', 'ph', 'sulphates', 'alcohol', 'quality'],
    outputCol="features")
iris_with_features = vecAssembler.transform(iris)