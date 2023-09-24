(**
It would not be a supervised learning demonstration without MNIST!
Here we show how simple linear logistic regression can get ~8% error, before showing how to improve 
train time using HOGWILD! Finally we show how you can get better performance using random Fourier Features.
*)

#r "nuget: MathNet.Numerics"
#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: MathNet.Numerics.MKL.Win"
#r "nuget: FSharp.Collections.ParallelSeq"
#r "nuget: FLearn, 2.1.2"

open System
open System.Diagnostics
open FSharp.Collections.ParallelSeq
open FLearn
open MathNet.Numerics.LinearAlgebra

//Helper Functions.

//This function will be used to assess train times
let timeFunction f x =
    let stopwatch = Stopwatch.StartNew()
    let result = f x
    stopwatch.Stop()
    printfn "Elapsed time: %f seconds" (stopwatch.Elapsed.TotalSeconds)
    result

(**
While not in Base FLearn, as it requires parallel sequences, HOGWILD! is a simple technique to parallelize Online convex optimization.
No locking here! Just iterate over the train sequence in parallel....
While it invalidates our online estimate of the batch loss, it does improve training time!
*)
let hogWild sample learner =
    PSeq.iter (fun s -> learn s learner) sample

//Data -- You will need to change this to point to a local version of the MNIST data set to replicate my results.
let path = @"C:\Users\User\OneDrive\Data\MNIST\"
let trainPath = path + "MNIST train.txt"
let testPath = path + "MNIST test.txt"

//Set the expected dimension of the input data and parse lib SVM format data to a vector and label.
let d = 784

let parse (s: string) =
    let blocks = s.TrimEnd(' ').Split(' ')
    let label = int blocks.[0]
    let featureVector = DenseVector.zero d

    for s in blocks.[1..] do
        let stuff = s.Split(':')
        let index = int stuff.[0] - 1
        let value = float32 stuff.[1]
        featureVector.[index] <- value

    (featureVector, label)

let train = IO.File.ReadLines(trainPath) |> Seq.map parse
let test = IO.File.ReadLines(testPath) |> Seq.map parse

//Multi-class Logistic Regression
let k = 10 // 10 classes
let loss = logLoss k

//First Order Features
let linearLearner = identityFeatureMap d |> createProjectedSGDLearner loss ignore
timeFunction (learnSequence train) linearLearner
let performanceLinear = PerformanceMeasures.loss01 test linearLearner

//First Order Features HOGWILD!

let linearLearnerHOGWILD = identityFeatureMap d |> createSGDLearner loss
timeFunction (hogWild train) linearLearnerHOGWILD

let performanceLinearLearnerHOGWILD =
    PerformanceMeasures.loss01 test linearLearnerHOGWILD

//Random Fourier Features

MathNet.Numerics.Control.UseNativeMKL() // Using MKL for Speed

(**
Random Fourier Features work by providing a randomized, finite dimensional approximation to the Gaussian feature map.
We wont go into the details here, suffice to say they are a powerful, non linear preprocessing trick
that one can use to increase performance over vanilla linear logistic regression.
*)
let randomFourierLearner =
    randomFourierFeatureMap 6.0f d 1000 |> createAdaGradLearner 1.0f loss // Using AdaGrad here as it yields better predictive performance.

timeFunction (hogWild train) randomFourierLearner

let performanceRandomFourierLearner =
    PerformanceMeasures.loss01 test randomFourierLearner // Note the increase in performance.
