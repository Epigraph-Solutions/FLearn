#r "nuget: FLearn"

open System
open FLearn
open FLearn.LearningWithCorruption
open MathNet.Numerics.LinearAlgebra

//Data -- You will need to change this to point to a local version of the Adult Census UCI data set to replicate my results"
let path = @"C:\Users\User\OneDrive\Data\Adult Data Set\"
let trainPath = path + "adult.txt"
let testPath = path + "adult test.txt"

//Parse SVM light format data to a vector and label
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

//Corrupted Training Data
let sigmaNegative = 0.1f
let sigmaPositive = 0.4f
let T = 
    [
        [1.0f - sigmaNegative ; sigmaPositive]
        [sigmaNegative ; 1.0f - sigmaPositive]
    ]
    |> matrix   
let R = Matrix.inverse T

let noise label : Distribution<CorruptedLabel> =
    distribution {
        match label with
        | 0 -> 
            let! z = Distribution.coinFlip sigmaNegative
            if z then return Corrupted(1,R)
            else return Corrupted(0, R)
        | _ -> 
            let! z = Distribution.coinFlip sigmaPositive
            if z then return Corrupted(0,R)
            else return Corrupted(1, R)
    }
let corrupt y = Distribution.sample <| noise y
let corruptedTrain = train |> Array.map (fun (x, y) -> (x, corrupt y))
let cleanTest = test |> Array.map (fun (x,y) -> x, Clean y)

//Corrupted Logistic Regression
let loss = logLossCC 2
let features = identityFeatureMap d
let learner = createSGDLearner loss features

learnSequence corruptedTrain learner

let performance = PerformanceMeasures.lossBatch (zeroOneLossCC 2) cleanTest learner