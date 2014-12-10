package main

import (
	"fmt"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/linear_models"
)

func main() {
	base.Silent()
	fmt.Println("Loading data...")
	trainData, err := base.ParseCSVToInstances("data/mnist_train.csv", true)
	if err != nil {
		panic(err)
	}
	testData, err := base.ParseCSVToInstances("data/mnist_test.csv", true)
	if err != nil {
		panic(err)
	}

	classifier, err := linear_models.NewLinearSVC("l1", "l2", true, 1.0, 1e-4)
	if err != nil {
		panic(err)
	}
	fmt.Println("Training...")
	classifier.Fit(trainData)

	fmt.Println("Predicting...")
	predictions, err := classifier.Predict(testData)
	if err != nil {
		panic(err)
	}

	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))
}
