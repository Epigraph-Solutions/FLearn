#r "nuget: FLearn"
#r "nuget: XPlot.Plotly"

open System
open FLearn
open MathNet.Numerics.LinearAlgebra
open XPlot.Plotly

//Helpers
let runningAverage (arr: float32 []) =
    let mutable sum = 0.0f
    let res = Array.zeroCreate<float32> arr.Length
    for i in 0 .. arr.Length - 1 do
        sum <- sum + arr.[i]
        res.[i] <- sum / float32 (i + 1)
    res

//Data -- You will need to change this to point to a local version of the Adult Census UCI data set to replicate my results
let path = @"C:\Users\User\OneDrive\Data\Adult Data Set\"
let trainPath = path + "adult.txt"

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
        featureVector.[d - 1] <- 1.0f

    (featureVector, label)
let train = 
    IO.File.ReadLines(trainPath) 
    |> Seq.map parse

//Online Logistic Regression
let loss = logLoss 2
let features = identityFeatureMap d
let learner = createSGDLearner loss features

let onlineRun = 
    train
    |> Seq.map(fun (x, y) ->
        let prediction = predictOnline learner x
        learn (x, y) learner
        lossIncurred loss y prediction
    )
    |> Seq.toArray
    |> runningAverage

let plot = Scatter(x = [0 .. onlineRun.Length - 1], y = onlineRun)
do Chart.Plot(plot).Show()

let performance = PerformanceMeasures.lossBatch loss train learner
let performanceOnline = onlineRun[onlineRun.Length - 1]