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

# classification - naive bayes
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

splits = iviris_df.randomSplit([0.6, 0.4], 1)
train_df = splits[0]
test_df = splits[1]
train_df.count()
test_df.count()
iviris_df.count()

nb = NaiveBayes(modelType='multinomial')
nbmodel = nb.fit(train_df)
predictions_df = nbmodel.transform(test_df)
predictions_df.take(1)

evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionsCol='prediction', metricName='accuracy')
nbaccuracy = evaluator.evaluate(predictions_df)
nbaccuracy


# multi-layer perceptron
iviris_df.take(1)
train_df.count()
test_df.count()
iviris_df.count()

from pyspark.ml.classificaiton import MultilayerPercptronClassifier
layers = [4, 5, 5, 3]

mlp = MultilayerPerceptronClassifier(layers=layers, seed=1)
mlp_model = mlp.fit(train_df)
mlp_predictions = mlp_model.transform(test_df)

mlp_evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
mlp_accuracy = mlp_evaluator.evaluate(mlp_predictions)
mlp_accuracy

# decision trees
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol='label', featureCol='features')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)

dt_evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
dt_accuracy = dt_evaluator.evaluate(dt_predictions)
dt_accuracy
