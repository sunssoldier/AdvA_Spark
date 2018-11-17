// cd Documents\Spark_Projects
// %spark%\bin\spark-shell

// spark-shell -i rain_forest.scala
// :load rain_forest.scala

//// Set preferences
sc.setLogLevel("ERROR")

//// Import data and rename columns

val dataWithoutHeader = spark.read.
	option("inferSchema", true).
	option("header", true).
	csv("covtype.data")

val colNames = Seq(
	"Elevation","Aspect","Slope",
	"Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
	"Horizontal_Distance_To_Roadways",
	"Hillshade_9am", "Hillshade_Noon","Hillshade_3pm",
	"Horizontal_Distance_Fire_Points"
	) ++ (
		(0 until 4).map( i => s"Wilderness_Area_$i")
	) ++ (
		(0 until 40).map( i => s"Soil_Type_$i")
	) ++ Seq("Cover_Type")

val data = dataWithoutHeader.toDF(colNames:_*).
	withColumn("Cover_Type", $"Cover_Type".cast("double"))

val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1))

trainData.cache()
testData.cache()

//// Assemble feature vectory

import org.apache.spark.ml.feature.VectorAssembler

val inputCols = trainData.columns.filter(_ != "Cover_Type")
val assembler = new VectorAssembler().
	setInputCols(inputCols).
	setOutputCol("featureVector")

val assembledTrainData = assembler.transform(trainData)

assembledTrainData.select("featureVector").show(truncate = false)

//// Build decision Tree

import org.apache.spark.ml.classification.DecisionTreeClassifier
import scala.util.Random

val classifier = new DecisionTreeClassifier().
	setSeed(Random.nextLong()).
	setLabelCol("Cover_Type").
	setFeaturesCol("featureVector").
	setPredictionCol("prediction")

val model = classifier.fit(assembledTrainData)
println(model.toDebugString)

model.featureImportances.toArray.zip(inputCols).
	sorted.reverse.foreach(println)