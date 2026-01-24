(**
Here we show how the ProjectedSGDLearner type can be used to perform a L1 regularized logistic regression.
*)

#r "nuget: FLearn, 2.1.2"

open System
open FLearn
open FLearn.PerformanceMeasures
open MathNet.Numerics.LinearAlgebra

//Data -- You will need to change this to point to a local version of the Adult Census UCI data set to replicate my results.
let path = @"C:\Users\brend\OneDrive\Data\Adult Data Set\"
let trainPath = path + "adult_train.svm"
let testPath = path + "adult_test.svm"

//Set the expected dimension of the input data and parse lib SVM format data to a vector and label.
let d = 105

let parse (s: string) =
    let blocks = s.TrimEnd(' ').Split(' ')

    let label =
        match blocks.[0] with
        | "-1" -> 0
        | "1" -> 1
        | _ -> failwith "Invalid Label"

    let featureVector = DenseVector.zero d
    featureVector.[d - 1] <- 1.0f

    for s in blocks.[1..] do
        let stuff = s.Split(':')
        let index = int stuff.[0]
        let value = float32 stuff.[1]
        featureVector.[index] <- value

    (featureVector, label)

let train = IO.File.ReadAllLines(trainPath) |> Array.map parse
let test = IO.File.ReadAllLines(testPath) |> Array.map parse

//Logistic Regression
let loss = logLoss 2
let features = identityFeatureMap d
let projection = Projections.L1Ball 10.0f // This ensures the weights used by our learner stay in an L1 Ball of radius 10.
let learner = createProjectedSGDLearner loss projection features

learnSequence train learner

//The optimizing over the L1 Ball promotes sparsity, in that most of the weights are close to zero. Here we demonstrate this.
learner.AverageWeights.CoerceZero(0.001)

learner.AverageWeights
|> Matrix.toSeqSkipZeros
|> Seq.length
|> printfn "Number of non-zero weights: %A"

let performance = loss01 test learner

printfn "Test Error: %A" performance
