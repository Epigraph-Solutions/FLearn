#r @"C:\Users\User\OneDrive\Code\FLearn\bin\Release\net7.0\FLearn.dll"
#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: XPlot.Plotly"

open System
open FLearn
open XPlot.Plotly
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions

MathNet.Numerics.Control.UseSingleThread()

//Helpers
let rnd = Distribution.rnd
let normal mean sigma = 
    Distribution.create (fun () -> Sample.normal mean sigma rnd)
let uniform lower upper = 
    Distribution.create (fun () -> Sample.continuousUniform lower upper rnd)
let toVector (x : float32) = DenseVector.raw [|x|]
let toFloat (v : Vector<float32>) = v.[0]

//Distribution
let Px = uniform -2.0 2.0
let trueMean x = 
    if x < -2.0 / 3.0 then 3. * (x + 1.0) 
    else if (x >= -2.0 / 3.0) && (x <= -1.0 / 3.0) then 1.0 + sin (30.0 * Math.PI * x / 2.0)
    else if (x >= -1.0 / 3.0) && (x <= 0.0) then -3.0 * x
    else sin (2.0 * Math.PI * x ) * sin (20.0 * Math.PI * x)
let noise x = 
    if x < 0.0 then  
        0.25 * cos(2.0 * Math.PI * x) |> abs 
    else if x > 0.0 && x < 1.0 then
        0.25
    else 
        0.1 * cos(8.0 * Math.PI * x) |> abs
let P = 
    distribution {
        let! x = Px
        let yMean = trueMean x
        let sigma = noise x
        let! y = normal yMean sigma
        return toVector(float32 x), float32 y
    }

//Sample
let n = 20000
let sample = 
    P
    |> Distribution.samples 
    |> Seq.take n 
    |> Seq.toArray

//Define training protocol
let train n sample learner =
    let S = Distribution.empirical sample
    learnDistribution S n learner
    tailAverage learner
    learnDistribution S n learner

//Setting up Candidate Feature Maps
let features sigma = 
    sample
    |> Array.map fst
    |> gaussianFeatureMap sigma
    |> GaussianTools.cover 0.1f
let featureMaps = Array.Parallel.map features [|0.01f .. 0.01f .. 0.05f|] 

//Online Learning -- Mean
let learnersMean = Array.map (createSGDLearner squareLoss) featureMaps
do Array.Parallel.iter (train 50000 sample) learnersMean
let mean = Array.minBy averageLoss learnersMean

//Online Learning -- Variance
let errors = PerformanceMeasures.lossesBatch squareLoss sample mean
let learnersVariance = Array.map (createSGDLearner errorLoss) featureMaps
do Array.Parallel.iter (train 50000 errors) learnersVariance
let variance = Array.minBy averageLoss learnersVariance

//Charting
let range = [|-2.0f .. 0.001f .. 2.0f|]
let dataPlot = 
    Scatter(
        x = Array.map (fst >> toFloat) sample, 
        y = Array.map snd sample, 
        mode= "markers", 
        name = "Data"
    )
let meanPlot = 
    Scatter(
        x = range, 
        y = Array.map (toVector >> mean.Predict >> standardLink) range, 
        marker = Marker(color = "DarkRed"),
        name = "Mean"
    )
let upperErrorPlot = 
    Scatter(
        x = range,
        y = Array.map (fun x -> let x = toVector x in standardLink(mean.Predict x) + 3.0f * sqrt(errorLink(variance.Predict x))) range,
        marker = Marker(color = "Red"),
        name = "Upper Bound"
    )
let lowerErrorPlot = 
    Scatter(
        x = range,
        y = Array.map (fun x -> let x = toVector x in standardLink(mean.Predict x) - 3.0f * sqrt(errorLink(variance.Predict x))) range,
        marker = Marker(color = "Red"),
        name = "Lower Bound"
    )
let plot = Chart.Plot([dataPlot ; meanPlot ; upperErrorPlot ; lowerErrorPlot])

do plot.Show()