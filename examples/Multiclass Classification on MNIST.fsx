(**
It would not be a supervised learning demonstration without MNIST!
Here we show how simple linear logistic regression can get ~8% error, before showing how to improve 
train time using HOGWILD! Finally we show how you can get better performance using random Fourier Features.
*)

#r "nuget: MathNet.Numerics"
#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: MathNet.Numerics.MKL.Win"
#r "nuget: FSharp.Collections.ParallelSeq"
#r @"C:\Users\User\OneDrive\Code\FLearn\bin\Release\net7.0\FLearn.dll"

open System
open System.Diagnostics
open FSharp.Collections.ParallelSeq  
open FLearn
open MathNet.Numerics.LinearAlgebra

//This function will be used to assess train times
let timeFunction f x =
    let stopwatch = Stopwatch.StartNew()
    let result = f x
    stopwatch.Stop()
    printfn "Elapsed time: %f seconds" (stopwatch.Elapsed.TotalSeconds)
    result

let rnd = new Random()

let sampleWithoutReplacement (n: int) (arr: 'a array)  =
    let mutable indices = List.init (Array.length arr) id
    let mutable result = []
    for _ in 1 .. n do
        let idx = rnd.Next(List.length indices)
        let chosen = List.item idx indices
        indices <- List.filter ((<>) chosen) indices
        result <- arr.[chosen] :: result
    result |> List.toArray

//Data -- You will need to change this to point to a local version of the MNIST data set to replicate my results.
let path = @"C:\Users\User\OneDrive\Data\MNIST\"
let trainPath = path + "MNIST train.txt"
let testPath = path + "MNIST test.txt"

//Set the expected dimension of the input data and parse lib SVM format data to a vector and label.
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

//Multi-class Logistic Regression
let k = 10 // 10 classes
let loss = logLoss k

//First Order Features
let linearLearner = 
    identityFeatureMap d
    |> createSGDLearner loss
timeFunction (learnSequence train) linearLearner
let performanceLinear = PerformanceMeasures.loss01 test linearLearner

//First Order Features HOGWILD!

(**
While not in Base FLearn, as it requires parallel sequences, HOGWILD! is a simple technique to parallelize Online convex optimization.
No locking here! Just iterate over the train sequence in parallel....
While it invalidates our online estimate of the batch loss, it does improve training time!
*)
let hogWild sample learner = PSeq.iter (fun s -> learn s learner) sample

let linearLearnerHOGWILD = 
    identityFeatureMap d
    |> createSGDLearner loss
timeFunction (hogWild train) linearLearnerHOGWILD
let performanceLinearLearnerHOGWILD = PerformanceMeasures.loss01 test linearLearnerHOGWILD

//Random Fourier Features

MathNet.Numerics.Control.UseNativeMKL() // Using MKL for Speed

(**
Random Fourier Features work by providing a randomized, finite dimensional approximation to the Gaussian feature map.
We wont go into the details here, suffice to say they are a powerful, non linear preprocessing trick
that one can use to increase performance over vanilla linear logistic regression.
*)
let randomFourierLearner =
    train
    |> Array.map fst
    |> Covers.gaussianKernelCoverParallel 1000 0.1f 6.0f
    |> nystromFeatureMap 2.0f 6.0f
    |> createAdaGradLearner 1.0f loss

randomFourierLearner.Dimension

timeFunction (hogWild train) randomFourierLearner
let performanceRandomFourierLearner = PerformanceMeasures.loss01 test randomFourierLearner //Note the increase in performance