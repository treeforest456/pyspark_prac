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

