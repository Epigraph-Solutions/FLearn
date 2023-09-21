(**
Here we demonstrate FLearns usage in the online setting, 
wherein data is learned from and predictions are assessed "as they come", rather than in a big batch.
*)

#r "nuget: FLearn"
#r "nuget: XPlot.Plotly"

open System
open FLearn
open MathNet.Numerics.LinearAlgebra
open XPlot.Plotly

//Helpers - This helper function calculates the running average of an array, useful for checking FLearns online performance.
let runningAverage (arr: float32 []) =
    let mutable sum = 0.0f
    let res = Array.zeroCreate<float32> arr.Length
    for i in 0 .. arr.Length - 1 do
        sum <- sum + arr.[i]
        res.[i] <- sum / float32 (i + 1)
    res

//Data -- You will need to change this to point to a local version of the Adult Census UCI data set to replicate my results.
let path = @"C:\Users\User\OneDrive\Data\Adult Data Set\"
let trainPath = path + "adult.txt"

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
        featureVector.[d - 1] <- 1.0f

    (featureVector, label)
let train = 
    IO.File.ReadLines(trainPath) 
    |> Seq.map parse

//Online Logistic Regression - Once Again we setup our loss, feature map and learner.
let loss = logLoss 2
let features = identityFeatureMap d
let learner = createSGDLearner loss features

//Here we setup our online run
let onlineRun = 
    train
    |> Seq.map (fun (x, y) (* The learner receives the pair (x,y) *)->
        let prediction = predictOnline learner x // It first makes a prediction using x.
        let l = lossIncurred loss y prediction // It then measures accuracy of this prediction against the true y.
        learn (x, y) learner // It then learns from its mistake.
        l // Finally we output the loss.
    )
    |> Seq.toArray
    |> runningAverage // Here we calculate the running average of the performance.

//Here we calculate the batch performance against the whole training set. Note that it is very close to the online performance of our learner.
let performance = PerformanceMeasures.lossBatch loss train learner
let performanceOnline = onlineRun[onlineRun.Length - 1]

//Finally, we plot our results, showing how the online performance improves with time, and approaches the batch performance.
[
    Scatter(x = [0 .. onlineRun.Length - 1], y = onlineRun) ;
    Scatter(x = [0 .. onlineRun.Length - 1], y = Seq.replicate onlineRun.Length performance)
]
|> Chart.Plot
|> Chart.Show