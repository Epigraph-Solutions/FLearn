#r "nuget: FLearn"

(**
Shameless self promotion incoming. Here we show how FLearn can learn classifiers from noisily labeled training sets,
utilizing techniques I had a hand in generalizing.
*)

open System
open FLearn
open FLearn.LearningWithCorruption
open MathNet.Numerics.LinearAlgebra

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

    featureVector.[d-1] <- 1.0f
    (featureVector, label)
let train = 
    IO.File.ReadAllLines(trainPath) 
    |> Array.map parse
let test =
    IO.File.ReadAllLines(testPath) 
    |> Array.map parse
    
//Corrupted Training Data - Here we define the noise rates on the two classes.
let sigmaNegative = 0.1f
let sigmaPositive = 0.4f
let T = 
    [
        [1.0f - sigmaNegative ; sigmaPositive]
        [sigmaNegative ; 1.0f - sigmaPositive]
    ]
    |> matrix   

//This Matrix is used for reconstructing the loss on the uncorrupted data from the loss on the corrupted data.
let R = Matrix.inverse T

//This defines the conditional distribution of noisy labels given clean labels.
let noise label : Distribution<CorruptedLabel> =
    distribution {
        match label with
        | 0 -> 
            let! flipped = Distribution.coinFlip sigmaNegative // sigmaNegative chance of Flipping a label 0 -> 1.
            if flipped then return Corrupted(1,R) // Note we return the noisy label AND the reconstruction that tells use how it relates to the clean label.
            else return Corrupted(0, R)
        | _ -> 
            let! flipped = Distribution.coinFlip sigmaPositive // sigmaPositive chance of Flipping a label 1 -> 0.
            if flipped then return Corrupted(0,R)
            else return Corrupted(1, R)
    }
let corrupt y = Distribution.sample <| noise y // This corrupts a clean label.
let corruptedTrain = train |> Array.map (fun (x, y) -> (x, corrupt y)) // This corrupts the training set.
let cleanTest = test |> Array.map (fun (x,y) -> x, Clean y) // This make sure we know the testing set is uncorrupted.

//Corrupted Logistic Regression.
let loss = logLossCC 2 // Note we now use the corruption corrected loss.
let features = identityFeatureMap d
let learner = createSGDLearner loss features

learnSequence corruptedTrain learner

let performance = PerformanceMeasures.lossBatch (zeroOneLossCC 2) cleanTest learner