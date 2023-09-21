(**
Here we start with a simple example, showing how FLearn can be used to learn a classifier for the UCI Adult Census Data set. We start by 
downloading the FLearn package from nuget, before opening the relevant libraries.
*)
#r "nuget: XPlot.Plotly"
#r "nuget: FLearn"

open System
open FLearn
open MathNet.Numerics.LinearAlgebra
open XPlot.Plotly

//Data -- You will need to change this to point to a local version of the Adult Census UCI data set to replicate my results.
let path = @"C:\Users\User\OneDrive\Data\Adult Data Set\"
let trainPath = path + "adult.txt"
let testPath = path + "adult test.txt"

//Set the expected dimension of the input data and parse lib SVM format data to a vector and label.
let d = 124
let parse (s : string) = 
    let blocks = s.TrimEnd(' ').Split(' ')
    let label = 
        match blocks.[0] with
        | "-1" -> 0
        | "+1" -> 1
        | _ -> failwith "Invalid Label"
    let featureVector = DenseVector.zero d

    for s in blocks.[1 ..] do
        let stuff = s.Split(':')
        let index = int stuff.[0] - 1
        let value = float32 stuff.[1]
        featureVector.[index] <- value
        featureVector.[d - 1] <- 1.0f

    (featureVector, label)

//Define the train and test sets, noting that data is NOT read into main memory. This is a feature of FLearn.
let train = 
    IO.File.ReadLines(trainPath) 
    |> Seq.map parse
let test =
    IO.File.ReadLines(testPath) 
    |> Seq.map parse

//Logistic Regression - Here we define the loss function, features and learner.
let loss = logLoss 2 // 2 for 2 classes, this is binary classification after all!
let features = identityFeatureMap d // The "do nothing" feature map, for data that is ready to go without processing.
let learner = createSGDLearner loss features // This sets the learning algorithm up, SGD in this case.

learnSequence train learner // Now we learn the training sequence using our learner

let performance = PerformanceMeasures.lossBatch (zeroOneLoss 2) test learner // Finally, we check the performance on the test set.

//Confusion and ROC curves
let confusionCurve = PerformanceMeasures.createConfusionCurve test learner //Creates curve of confusion matrices as we vary the threshold over which we classify as 1.
let truePostiveRates, trueNegativeRates, thresholds = Array.unzip3 confusionCurve.RocCurve // Extracts true positive and true negative rates from ROC curve as well as thresholds.

Scatter(x = truePostiveRates, y = trueNegativeRates, text = thresholds, hoverinfo="text")
|> Chart.Plot
|> Chart.Show