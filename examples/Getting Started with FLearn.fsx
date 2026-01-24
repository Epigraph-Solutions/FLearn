(**
Here we start with a simple example, showing how FLearn can be used to learn a classifier for the UCI Adult Census Data set. We start by 
downloading the FLearn package from nuget, before opening the relevant libraries.
*)

#r @"../bin/Release/net7.0/FLearn.dll"
#r "nuget: MathNet.Numerics.FSharp"

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
//Set the cost for false positives.

let parse (s: string) =
    let blocks = s.TrimEnd(' ').Split ' '

    let label =
        match blocks.[0] with
        | "-1" -> Negative
        | "1" -> Positive
        | _ -> failwith "Invalid Label"

    let featureVector = DenseVector.zero d
    featureVector.[d - 1] <- 1.0f

    for s in blocks.[1..] do
        let stuff = s.Split ':'
        let index = int stuff.[0]
        let value = float32 stuff.[1]
        featureVector.[index] <- value

    featureVector, label

//Define the train and test sets, noting that data is NOT read into main memory. This is a feature of FLearn.
let train = IO.File.ReadLines trainPath |> Seq.map parse
let test = IO.File.ReadLines testPath |> Seq.map parse


//Binary Classification - Here we define the loss function, features and learner.
let loss = canonicalHinge //The canonical hinge loss is appropriate for binary classification.
let features = identityFeatureMap d // The "do nothing" feature map, for data that is ready to go without processing.
let learner = createSGDLearner loss features // This sets the learning algorithm up, SGD in this case.

learnSequence train learner // Now we learn the training sequence using our learner.

//Evaluate performance on the test set.
let performance = lossBatch loss test learner
printfn "0/1 Loss Performance: %f" performance
