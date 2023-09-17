#r "nuget: FLearn"
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
let Px = uniform -Math.PI Math.PI
let trueMean x = 
    if x < 0.0 then 2.0 * x
    else sin (2.0 * x)
let noise x = 0.75 * cos(2.0 * x) |> abs
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

//Setting up Candidate Feature Maps
let features sigma = 
    sample
    |> Array.map fst
    |> gaussianFeatureMap sigma
    |> GaussianTools.cover 0.2f
let featureMaps = Array.Parallel.map features [|0.1f .. 0.01f .. 0.4f|] 

//Online Learning -- Mean
let learnersMean = Array.map (createSGDLearner squareLoss) featureMaps
do Array.Parallel.iter (learnSequence sample) learnersMean
let mean = Array.minBy averageLoss learnersMean

//Online Learning -- Variance
let errors = PerformanceMeasures.lossesBatch squareLoss sample mean
let learnersVariance = Array.map (createSGDLearner positiveRegressionLoss) featureMaps
do Array.Parallel.iter (learnSequence errors) learnersVariance
let variance = Array.minBy averageLoss learnersVariance

//Charting
let range = [|-float32 Math.PI .. 0.001f .. float32 Math.PI|]

[
    Scatter(
        x = Array.map (fst >> toFloat) sample, 
        y = Array.map snd sample, 
        mode= "markers", 
        name = "Data"
    );
    Scatter(
        x = range, 
        y = Array.map (toVector >> mean.Predict >> standardLink) range, 
        marker = Marker(color = "DarkRed"),
        name = "Mean"
    );
    Scatter(
        x = range,
        y = Array.map (fun x -> let x = toVector x in standardLink(mean.Predict x) + 3.0f * sqrt(positiveRegressionLink(variance.Predict x))) range,
        marker = Marker(color = "Red"),
        name = "Upper Bound"
    );
    Scatter(
        x = range,
        y = Array.map (fun x -> let x = toVector x in standardLink(mean.Predict x) - 3.0f * sqrt(positiveRegressionLink(variance.Predict x))) range,
        marker = Marker(color = "Red"),
        name = "Lower Bound"
    )
]
|> Chart.Plot
|> Chart.Show