#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: MathNet.Numerics.MKL.Win-x64"
#r "nuget: FLearn"

open System
open FLearn
open MathNet.Numerics.LinearAlgebra

//Data -- You will need to change this to point to a local version of the Adult Census UCI data set to replicate my results
let path = @"C:\Users\User\OneDrive\Data\Adult Data Set\"
let trainPath = path + "adult.txt"
let testPath = path + "adult test.txt"

//Parse SVM light format data to a vector and label
let d = 1000000000
let parse (s : string) = 
    let blocks = s.TrimEnd(' ').Split(' ')
    let label = 
        match blocks.[0] with
        | "-1" -> 0
        | "+1" -> 1
        | _ -> failwith "Invalid Label"
    let featureVector = SparseVector.zero d

    for s in blocks.[1 ..] do
        let stuff = s.Split(':')
        let index = int stuff.[0] - 1
        let value = float32 stuff.[1]
        featureVector.[index] <- value
        featureVector.[d - 1] <- 1.0f

    (featureVector, label)
let train = 
    IO.File.ReadLines(trainPath) 
    |> Seq.map parse
let test =
    IO.File.ReadLines(testPath) 
    |> Seq.map parse

//Multiclass Logistic Regression---------------------------------------------------------------------------------------------
let k = 2
let loss = logLoss k

//First Order Features
let linearLearner = 
    identityFeatureMap d
    |> SparseLearning.createSparseSGDLearnerNoAverage loss

learnSequence train linearLearner

let performanceLinear = PerformanceMeasures.lossBatch (zeroOneLoss k) test linearLearner