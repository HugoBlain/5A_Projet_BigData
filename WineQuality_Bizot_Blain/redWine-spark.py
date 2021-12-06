# Doc Kmeans :
# --> https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.KMeans.html

# DataFrame : 
# --> Un DataFrame est une collection distribuée de données organisées en colonnes nommées. Il est conceptuellement équivalent à une table dans une base de données relationnelle ou à un bloc de données en R/Python, mais avec des optimisations plus riches sous le capot.
# --> https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html


import matplotlib.pyplot as plt
import numpy as np
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
    StructField("id", FloatType(), nullable),
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


# connection à la base de données
sqlContext = SQLContext(sparkContext = spark.sparkContext, sparkSession = spark)


# Charger les données en dataFrame spark
redWine_dataFrame = sqlContext.read.csv('winequality-red.csv', sep=";", header = False, schema = schema)


# afficher dataframe initial
print("\nDataFrame initial (", redWine_dataFrame.select("*").count(), ") lignes :")
redWine_dataFrame.select("*").limit(5).show()


# création de la colonne "features" (assemblage des données numériques)
# --> inputs = toutes sauf "id" et "quality"
# --> outputs = "features"
vecAssembler = VectorAssembler(
    inputCols=['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide','density', 'ph', 'sulphates', 'alcohol'],
    outputCol="features"
)


# pour la suite nous n'avons plus besoin que de 3 colonnes : "id", "quality", "features"
dataFrame_KMeans = vecAssembler.transform(redWine_dataFrame).select("id", "quality", "features")


# afficher le dataFrame avec la nouvelle colonne "features" 
print("\nDataFrame utilisé pour le KMeans :")
dataFrame_KMeans.select("*").limit(5).show()


# K-means
# --> utilisation de la colonne "features"
# --> la qualité est représentée par une note entre 0 et 10 (la prédiction sera donnée entre 0 et k)
k = 10  
kmeans_algo = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans_algo.fit(dataFrame_KMeans)
centers = model.clusterCenters()


# Coordonnées des centres des clusters
print("\nCentres des clusters :")
for center in centers:
    print(center)


# assigner chaque ligne (chaque vin) à un cluster
# --> pour la suite nous n'avons plus besoin de la colonne "features"
dataFrame_prediction = model.transform(dataFrame_KMeans).select("id", "quality", "prediction")


# afficher dataFrame avec la prédiction
print("\nDataFrame avec les prédictions :")
dataFrame_prediction.select("*").limit(5).show()


# convertir le dataFrame Spark en dataFrame Pandas
pandas_dataFrame = dataFrame_prediction.toPandas()


# vérifier si K-Means a su retrouver la qualité 
# --> attention, la prédiction n'est pas une note, c'est juste le numéro du cluster assigné

tab = np.zeros((11, 11))

for index, row in pandas_dataFrame.iterrows():
    # recupérer les infos
    q = int(row["quality"])
    p = int(row["prediction"])
    # tableau pour visualiser la répartition
    tab[q, p] += 1

print("Nombre de vin noté 6 : ", tab[6].sum())
print("La pluspart ont été rangé dans le cluster n ", np.where(tab[6] == tab[6].max()), " soit ", tab[6].max(), " vins")



fig = plt.figure(figsize=(11, 11))
plt.imshow(tab)
plt.title("Quality / Prédiction")
plt.colorbar()
plt.xlabel("Quality")
plt.ylabel("Prediction")
plt.show()


"""
for index, row in pandas_dataFrame.iterrows():

    quality = np.zeros(11)
    prediction = np.zeros(11)

    # recupérer les infos
    q = int(row["quality"])
    p = int(row["prediction"])
    # compteur le nombre de vins dans chaque qualité et clusters
    quality[q] += 1
    prediction[p] += 1
    # tableau pour visualiser la répartition
    tab[q, p] += 1

print("\nQualité " , quality)
print("Predict " , prediction, "\n\n")
"""







"""


#visualisation
high_qualityI = redWine_for_viz['quality'] == 7
high_quality = redWine_for_viz [ high_qualityI ]

#couleurs
colors = {0:'red', 1:'green', 2:'blue'}

fig = plt.figure().gca(projection='3d')
fig.scatter(high_quality.ph,
             high_quality.chlorides,
             high_quality.alcohol,
             c = high_quality.prediction.map(colors),
             marker = 's')


fig.set_xlabel('ph')
fig.set_ylabel('chlorides')
fig.set_zlabel('alcohol')
plt.show()

print("--> Fin")
"""