# archive.ics.uci.edu/ml/machine-learning-databases/00294/

from pyspark.ml.regression import LinearRegression
pp_df = spark.read.csv('CCPP/Folds5x2_pp_sheet1.csv', header=True, inferSchema=True)

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=['AT', 'V', 'AP', 'RH'], outputCol='features')
vpp_df = vectorAssembler.transform(pp_df)
vpp_df.take(1)

lr = LinearRegression(featuresCol='features', labelCol='PE')
lr_model = lr.fit(vpp_df)

lr_model.coefficients
lr_model.intercept

lr_model.summary.rootMeanSquaredError

lr_model.save('lr1.model')


# decision tree regressor
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

pp_df.take(1)
vectorAssembler = VectorAssembler(inputCols=['AT', 'V', 'AP', 'RH'], outputCol='features')
vpp_df = vectorAssembler.transform(pp_df)
vpp_df.take(1)

splits = vpp_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]
train_df.count()
test_df.count()
vpp_df.count()

dt = DecisionTreeRegressor(featuresCol='features', labelCol='PE')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(test_df)

dt_evaluator = RegressionEvaluator(labelCol='PE', predictionCol='prediction', metricName='rmse')
rmse = dt_evaluator.evaluate(dt_predictions)

