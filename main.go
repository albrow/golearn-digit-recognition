package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/linear_models"
)

func main() {
	// Load and parse the data from csv files
	fmt.Println("Loading data...")
	trainData, err := base.ParseCSVToInstances("data/mnist_train.csv", true)
	if err != nil {
		panic(err)
	}
	testData, err := base.ParseCSVToInstances("data/mnist_test.csv", true)
	if err != nil {
		panic(err)
	}

	// Create a new linear SVC with some good default values
	classifier, err := linear_models.NewLinearSVC("l1", "l2", true, 1.0, 1e-4)
	if err != nil {
		panic(err)
	}

	// Don't output information on each iteration
	base.Silent()

	// Train the linear SVC
	fmt.Println("Training...")
	classifier.Fit(trainData)

	// Make predictions for the test data
	fmt.Println("Predicting...")
	predictions, err := classifier.Predict(testData)
	if err != nil {
		panic(err)
	}

	// Get a confusion matrix and print out some accuracy stats for our predictions
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))
}
