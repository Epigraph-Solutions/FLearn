(**
Shameless self promotion incoming. Here we show how FLearn can learn classifiers from noisily labeled training sets,
utilizing techniques I had a hand in generalizing.
*)

#r "nuget: FLearn, 2.1.2"

open System
open FLearn
open FLearn.PerformanceMeasures
open FLearn.LearningWithCorruption
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

//Corrupted Training Data - Here we define the noise rates on the two classes.
let sigmaNegative = 0.2f
let sigmaPositive = 0.3f

let T =
    [ [ 1.0f - sigmaNegative; sigmaPositive ]
      [ sigmaNegative; 1.0f - sigmaPositive ] ]
    |> matrix

//This Matrix is used for reconstructing the loss on the uncorrupted data from the loss on the corrupted data.
let R = Matrix.inverse T

//This defines the conditional distribution of noisy labels given clean labels.
let noise label : Distribution<CorruptedLabel> =
    distribution {
        match label with
        | 0 ->
            let! flipped = Distribution.coinFlip sigmaNegative // sigmaNegative chance of Flipping a label 0 -> 1.

            if flipped then
                return Corrupted(1, R) // Note we return the noisy label AND the reconstruction that tells use how it relates to the clean label.
            else
                return Corrupted(0, R)
        | _ ->
            let! flipped = Distribution.coinFlip sigmaPositive // sigmaPositive chance of Flipping a label 1 -> 0.

            if flipped then
                return Corrupted(0, R)
            else
                return Corrupted(1, R)
    }

let corrupt y = Distribution.sample <| noise y // This corrupts a clean label.
let corruptedTrain = train |> Array.map (fun (x, y) -> (x, corrupt y)) // This corrupts the training set.

//Corrupted Logistic Regression.
let cleanLoss = logLoss 2
let ccLoss = logLossCC 2 // Note we now use the corruption corrected loss.
let features = identityFeatureMap d
let learner = createAdaGradLearner 0.1f ccLoss features

learnSequence corruptedTrain learner

//Define a shiny new predictor from the one learned with corrupted data.
let predictor =
    { new IPredictor<_, _> with
        member this.Predict x =
            learner.Predict x |> cleanLoss.LinkFunction }

let performance = loss01 test predictor

printfn "Corrupted Logistic Regression 0/1 Loss: %f" performance
