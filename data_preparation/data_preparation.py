from pyspark.ml.feature import MinMaxScaler, StandardScaler
from pyspark.ml.linalg import Vectors


# min max scaler
features_df = spark.createDataFrame([
	(1, Vectors.dense([10.0, 10000.0, 1.0]),),
	(2, Vectors.dense([20.0, 30000.0, 2.0]),),
	(3, Vectors.dense([30.0, 40000.0, 3.0]),)
], ['id', 'features'])

features_df.take(1)

feature_scaler = MinMaxScaler(inputCol='features', outputCol='sfeatures')
smodel = feature_scaler.fit(features_df)
sfeatures_df = smodel.transform(features_df)

sfeatures_df.take(1)

sfeatures_df.select('features', 'sfeatures').show()

# standard scaler
feature_stand_scaler = StandardScaler(inputCol='features',\
										outputCol='sfeatures',\
										withStd=True,\
										withMean=True)
stand_smodel = feature_stand_scaler.fit(features_df)
stand_sfeatures_df = stand_smodel.transform(features_df)
stand_sfeatures_df.take(1)
stand_sfeatures_df.show()

# bucketizer
from pyspark.ml.feature import Bucketizer
splits = [-float('inf'), -10.0, 0.0, 10.0, float('inf')]
b_data = [(-800.0,), (-10.5,), (-1.7,), (0.0,), (8.2,), (90.1,)]
b_df = spark.createDataFrame(b_data, ['features'])
b_df.show()

bucketizer = Bucketizer(splits=splits, inputCol='features', outputCol='bfeatures')
bucketed_df = bucketizer.transform(b_df)
bucketed_df.show()
