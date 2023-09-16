#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: MathNet.Numerics.MKL.Win-x64"
#r "nuget: FLearn"

open System
open FLearn
open MathNet.Numerics.LinearAlgebra

//MKL Path -- You will need to change this to your local MKL installation
MathNet.Numerics.Control.NativeProviderPath <- @"C:\MKL"
MathNet.Numerics.Control.UseNativeMKL()

//Data -- You will need to change this to point to a local version of the MNIST data set to replicate my results
let path = @"C:\Users\User\OneDrive\Data\MNIST\"
let trainPath = path + "MNIST train.txt"
let testPath = path + "MNIST test.txt"

//Parse Data
let d = 784
let parse (s : string) = 
    let blocks = s.TrimEnd(' ').Split(' ')
    let label = int blocks.[0]
    let featureVector = DenseVector.zero d
    
    for s in blocks.[1 ..] do
        let stuff = s.Split(':')
        let index = int stuff.[0] - 1
        let value = float32 stuff.[1]
        featureVector.[index] <- value

    (featureVector, label)
let train = 
    IO.File.ReadAllLines(trainPath) 
    |> Array.map parse
let test =
    IO.File.ReadAllLines(testPath) 
    |> Array.map parse

//Multiclass Logistic Regression---------------------------------------------------------------------------------------------
let k = 10
let loss = logLoss k

//First Order Features
let linearLearner = 
    identityFeatureMap d
    |> createSGDLearner loss

learnSequenceVerbosely train linearLearner

let performanceLinear = PerformanceMeasures.lossBatch (zeroOneLoss k) test linearLearner

//Second Order Features
let secondOrderLearner = 
    randomPolynomialMap 2 d 500
    |> createSGDLearner loss 

learnSequenceVerbosely train secondOrderLearner

let performanceSecondOrder = PerformanceMeasures.lossBatch (zeroOneLoss k) test secondOrderLearner