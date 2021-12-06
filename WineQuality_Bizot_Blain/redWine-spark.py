# Doc Kmeans :
# --> https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.clustering.KMeans.html

# DataFrame : 
# --> Un DataFrame est une collection distribuée de données organisées en colonnes nommées. Il est conceptuellement équivalent à une table dans une base de données relationnelle ou à un bloc de données en R/Python, mais avec des optimisations plus riches sous le capot.
# --> https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html

# Optimisation du choix du k
# --> https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb


from pyspark.ml import clustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml.clustering import ClusteringSummary, KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType, IntegerType
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.evaluation import ClusteringEvaluator


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
print("\n\nDataFrame initial (", redWine_dataFrame.select("*").count(), ") lignes :")
redWine_dataFrame.select("*").limit(5).show()



print("\n\n1ère partie : Visualisation 3D :\n##################################################################################################\n")

# création de la colonne "features" (assemblage des données numériques)
# --> inputs = 'fixed_acidity', 'ph', 'alcohol''
# --> outputs = "features"
vecAssembler = VectorAssembler(
    inputCols=['fixed_acidity', 'ph', 'alcohol'],
   outputCol="features")
redWine_with_features = vecAssembler.transform(redWine_dataFrame).select('fixed_acidity', 'ph', 'alcohol', 'features')


# optimisation du k choisis : on teste avec plusieurs valeur de k et on regarde laqeulle donne le meilleur résultat
# --> technique de la silhouette
k_max = 10
k_min = 2
silhouette_list = []
k_list = np.arange(k_min, k_max + 1)

print("Début recherche du k optimal : ")
for k in range(k_min, k_max + 1):
    # entrainement
    kmeans_algo = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans_algo.fit(redWine_with_features)
    centers = model.clusterCenters()
    # faire les prédictions
    prediction = model.transform(redWine_with_features)
    # evaluation des clusters en calculant le score de la silhouette
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(prediction)
    silhouette_list.append(silhouette)
    print("    Score pour k =", k, ":", silhouette)

# afficher les différentes valeurs de k et leur score
plt.plot(k_list, silhouette_list)
plt.xlabel("K")
plt.ylabel("Silhoutte Score")
plt.show()
     

# K-means
k = 2
kmeans_algo = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans_algo.fit(redWine_with_features)
centers = model.clusterCenters()


# afficher le dataFrame avec la nouvelle colonne "features" 
print("\nDataFrame utilisé pour le KMeans :")
redWine_with_features.select("*").limit(5).show()


# Coordonnées des centres des clusters
print("\nCentres des clusters :")
for center in centers:
    print(center)


# assigner chaque ligne (chaque vin) à un cluster (prediction)
# --> pour la suite nous n'avons plus besoin de la colonne "features"
redWine_with_clusters = model.transform(redWine_with_features).select('fixed_acidity', 'ph', 'alcohol', 'prediction')


# afficher dataFrame avec la prédiction
print("\nDataFrame avec les prédictions :")
redWine_with_clusters.select("*").limit(5).show()


# convertir le dataFrame Spark en dataFrame Pandas
redWine_for_viz = redWine_with_clusters.toPandas()


#couleurs
colors = {0:'red', 1:'green', 2:'blue', 3:'purple', 4:'yellow'}


fig = plt.figure().gca(projection='3d')
fig.scatter(redWine_for_viz.fixed_acidity,
             redWine_for_viz.ph,
             redWine_for_viz.alcohol,
             c = redWine_for_viz.prediction.map(colors),
             marker = 's'
            )

fig.set_xlabel('fixed_acidity')
fig.set_zlabel('ph')
fig.set_ylabel('alcohol')
plt.show()



print("\n2ème partie : Recherche de la qualité\n##################################################################################################\n")


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

quality = np.zeros(11)
prediction = np.zeros(11)

for index, row in pandas_dataFrame.iterrows():
    # recupérer les infos
    q = int(row["quality"])
    p = int(row["prediction"])
    # compteur
    quality[q] += 1
    prediction[p] += 1
    # tableau pour visualiser la répartition
    tab[p, q] += 1


print("Nombre de vin noté 6 : ", tab[:,6].sum())
print("Nombre de vin dans le cluster 6 : ", tab[6].sum())

print("\nQualité : ", quality)
print("\nPrédict : ", prediction, "\n")

fig = plt.figure(figsize=(11, 11))
plt.imshow(tab)
plt.title("Quality / Prédiction")
plt.colorbar()
plt.xlabel("Quality")
plt.ylabel("Prediction")
plt.show()



