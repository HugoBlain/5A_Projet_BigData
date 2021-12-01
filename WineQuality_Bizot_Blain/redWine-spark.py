# Doc Kmeans :
# --> https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.KMeans.html


import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType, IntegerType
from pyspark.mllib.linalg.distributed import RowMatrix


# lancement de spark
spark = SparkSession\
        .builder\
        .appName("RedWine_App")\
        .getOrCreate()


# déclaration de StructType
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


# connection BD
sqlContext = SQLContext(sparkContext = spark.sparkContext, sparkSession = spark)
# chargement des données du dataSet sous forme de dataFrame
# --> Un DataFrame est une collection distribuée de données organisées en colonnes nommées. Il est conceptuellement équivalent à une table dans une base de données relationnelle ou à un bloc de données en R/Python, mais avec des optimisations plus riches sous le capot.
redWineQuality = sqlContext.read.csv('winequality-red.csv', sep=";", header = False, schema = schema)


# afficher les colonnes
#print(redWineQuality.columns)
# afficher les données
#redWineQuality.select("*").show()




### répartition en colonne
#vecAssembler = VectorAssembler(
#    inputCols=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide','density', 'ph', 'sulphates', 'alcohol', 'quality'],
#    outputCol="features")
#redWineQuality_with_features = vecAssembler.transform(redWineQuality)




# K-means
#k = 3
#kmeans_algo = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
#model = kmeans_algo.fit(redWineQuality_with_features)
#centers = model.clusterCenters()
# clusters
#redWineQuality_with_clusters = model.transform(redWineQuality_with_features)

print("--> Fin")