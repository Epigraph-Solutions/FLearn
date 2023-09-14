#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: MathNet.Numerics.MKL.Win-x86"
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

let performanceLinear = PerformanceMeasures.lossSummary [|loss ; zeroOneLoss k|] test linearLearner
linearLearner.AverageLoss
performanceLinear.[0]
performanceLinear.[1]

//Second Order Features
let secondOrderLearner = 
    randomPolynomialMap 2 d 500
    |> createSGDLearner loss

let S = Distribution.empirical train

learnDistributionVerbosely S 10000 secondOrderLearner
tailAverage secondOrderLearner
learnDistributionVerbosely S 10000 secondOrderLearner

let performanceSecondOrder = PerformanceMeasures.lossSummary [|loss ; zeroOneLoss k|] test secondOrderLearner
secondOrderLearner.AverageLoss
performanceSecondOrder.[0]
performanceSecondOrder.[1]

//Random Fourier Features
let sigma = 6.0f
let randomFourierFeatures = randomFourierFeatureMap sigma d 500
let randomFourierLearner = createAdaGradLearner 1.0f loss randomFourierFeatures

learnDistribution S 10000 randomFourierLearner
tailAverage randomFourierLearner
learnDistribution S 10000 randomFourierLearner

let randomFourierPerformance = PerformanceMeasures.lossSummary [|loss ; zeroOneLoss k|] test randomFourierLearner
randomFourierLearner.AverageLoss
randomFourierPerformance.[0]
randomFourierPerformance.[1]