from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer

iris_df = spark.read.csv('iris.txt', inferSchema=True)
iris_df.take(1)

iris_df = iris_df.select(col('_c0').alias('sepal_length'),
	col('_c1').alias('sepal_width'),
	col('_c2').alias('petal_length'),
	col('_c3').alias('petal_width'),
	col('_c4').alias('species')
	)

iris_df.take(1)

vectorAssembler = VectorAssember(inputCols=['sepal_length', 'sepal_length', 'pedal_length', 'pedal_width'],\
	outputCol='features')

viris_df = vectorAssembler.transform(iris_df)

viris_df.take(1)

indexer = StringIndexer(inputCol='species', outputCol='label')
iviris_df = indexer.fit(viris_df).transform(viris_df)
iviris_df.show(1)

