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

# Init spark
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, StringType
from pyspark.mllib.linalg.distributed import RowMatrix

spark = SparkSession\
        .builder\
        .appName("MyTest")\
        .getOrCreate()
sqlContext = SQLContext(sparkContext = spark.sparkContext, sparkSession = spark)


# Load Iris data
nullable = True
schema = StructType([
    StructField("sepal_l", FloatType(), nullable),
    StructField("sepal_w", FloatType(), nullable),
    StructField("petal_l", FloatType(), nullable),
    StructField("petal_w", FloatType(), nullable),
    StructField("species", StringType(), nullable),
])
iris = sqlContext.read.csv('iris.data', header = False, schema = schema)

# Create features column, assembling together the numeric data
vecAssembler = VectorAssembler(
    inputCols=['sepal_l', 'sepal_w', 'petal_l', 'petal_w'],
    outputCol="features")
iris_with_features = vecAssembler.transform(iris)


# Do K-means
k = 3 # TODO: test several k, elbow method
kmeans_algo = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans_algo.fit(iris_with_features)
centers = model.clusterCenters()
# Assign clusters to flower
iris_with_clusters = model.transform(iris_with_features)

print("Centers", centers)

# Convert Spark Data Frame to Pandas Data Frame
iris_for_viz = iris_with_clusters.toPandas()

# Vizualize
# Marker styles are calculated from Iris species
# Marker styles are calculated from Iris species
setosaI = iris_for_viz['species'] == 'Iris-setosa'
setosa = iris_for_viz [ setosaI ]
versicolorI = iris_for_viz['species'] == 'Iris-versicolor'
versicolor = iris_for_viz [ versicolorI ]
virginicaI = iris_for_viz['species'] == 'Iris-virginica'
virginica = iris_for_viz [ virginicaI ]

# Colors code k-means results, cluster numbers
colors = {0:'red', 1:'green', 2:'blue'}

fig = plt.figure().gca(projection='3d')
fig.scatter(virginica.sepal_l,
             virginica.sepal_w,
             virginica.petal_l,
             c = virginica.prediction.map(colors),
             marker = 's')
fig.scatter(versicolor.sepal_l,
            versicolor.sepal_w,
            versicolor.petal_l,
            c = versicolor.prediction.map(colors),
            marker = 'v')
fig.scatter(setosa.sepal_l,
             setosa.sepal_w,
             setosa.petal_l,
             c = setosa.prediction.map(colors),
             marker = 'o')
fig.set_xlabel('sepal l')
fig.set_ylabel('sepal w')
fig.set_zlabel('petal l')
plt.show()

# Dimenstion reduction. From 4D to 3D
# by PCA method
datamatrix =  RowMatrix(iris.select(['sepal_l', 'sepal_w', 'petal_l', 'petal_w']).rdd.map(list))

# Compute the top 3 principal components. The "best" hyperplane.
pc = datamatrix.computePrincipalComponents(3)
print ("***** 3 Principal components *****")
print(pc)

# project data
projected = datamatrix.multiply(pc)
new_X = pd.DataFrame(
    projected.rows.map(lambda x: x.values[0]).collect()
)
new_Y = pd.DataFrame(
    projected.rows.map(lambda x: x.values[1]).collect()
)
new_Z = pd.DataFrame(
    projected.rows.map(lambda x: x.values[2]).collect()
)

# Vizualize with PCA, 3 components
# Colors code k-means results, cluster numbers
colors = {0:'red', 1:'green', 2:'blue'}

fig = plt.figure().gca(projection='3d')
fig.scatter(new_X [virginicaI],
            new_Y [virginicaI],
            new_Z [virginicaI],
            c = virginica.prediction.map(colors),
            marker = 's')
fig.scatter(new_X [versicolorI],
            new_Y [versicolorI],
            new_Z [versicolorI],
            c = versicolor.prediction.map(colors),
            marker = 'v')
fig.scatter(new_X [setosaI],
            new_Y [setosaI],
            new_Z [setosaI],
            c = setosa.prediction.map(colors),
            marker = 'o')
fig.set_xlabel('Component 1')
fig.set_ylabel('Component 2')
fig.set_zlabel('Component 3')
plt.show()


# Dimenstion reduction. From 4D to 2D
# by PCA method
# Compute the top 3 principal components. The "best" hyperplane.
pc = datamatrix.computePrincipalComponents(2)
print ("***** 2 Principal components. The same as first 2 of 3 principal components *****")
print (pc)

# project data
projected = datamatrix.multiply(pc)
new_X = pd.DataFrame(
    projected.rows.map(lambda x: x.values[0]).collect()
)
new_Y = pd.DataFrame(
    projected.rows.map(lambda x: x.values[1]).collect()
)

# Vizualize with PCA, 2 components
# Colors code k-means results, cluster numbers
colors = {0:'red', 1:'green', 2:'blue'}
fig = plt.figure().gca()
fig.scatter(new_X [virginicaI],
            new_Y [virginicaI],
            c = virginica.prediction.map(colors),
            marker = 's')
fig.scatter(new_X [versicolorI],
            new_Y [versicolorI],
            c = versicolor.prediction.map(colors),
            marker = 'v')
fig.scatter(new_X [setosaI],
            new_Y [setosaI],
            c = setosa.prediction.map(colors),
            marker = 'o')
fig.set_xlabel('Component 1')
fig.set_ylabel('Component 2')
plt.show()
