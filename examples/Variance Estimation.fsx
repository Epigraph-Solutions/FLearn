(**
One strength of Gaussian Proceses is that they provide both an estimate of the conditional mean, and the conditional Variance.
Here we show how one can achieve this using a model that is trained on the errors of model trained to find the conditional mean.
Along the way we utlize a spares approximation of the Gaussian kernel feature map.

One issue with doing regression on errors, is that errors are always positive! We bake this requirement into the loss used for doing the error 
estimation.
*)

#r "nuget: FLearn"
#r "nuget: XPlot.Plotly"

open System
open FLearn
open XPlot.Plotly
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions

//Helpers -- Just some distributions and conversions between vectors and floats.
let rnd = Distribution.rnd
let normal mean sigma = 
    Distribution.create (fun () -> Sample.normal mean sigma rnd)
let uniform lower upper = 
    Distribution.create (fun () -> Sample.continuousUniform lower upper rnd)
let toVector (x : float32) = DenseVector.raw [|x|]
let toFloat (v : Vector<float32>) = v.[0]

//Target Distribution, note the non uniformity of the error
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

//Sample form Target Distribution.
let n = 20000
let sample = 
    P
    |> Distribution.samples 
    |> Seq.take n 
    |> Seq.toArray

//Setting up Candidate Feature Maps
let features sigma = 
    sample
    |> Array.map fst // Extract x from the sample
    |> gaussianFeatureMap sigma // Create a Gaussian Feature map, \phi(x) = (K(x_1, x), K(x_2, x), ... , K(x_n, x)) with K the Gaussian kernel
    |> GaussianTools.cover 0.2f // Rather than using all of the sample in our gaussian feature map, we use a sparse representation. Magic :)
let featureMaps = Array.Parallel.map features [|0.1f .. 0.01f .. 0.4f|] //Setup the Feature Maps

//Online Learning -- Mean
let learnersMean = Array.map (createSGDLearner squareLoss) featureMaps // Setup candidate learners for the conditional mean
Array.Parallel.iter (learnSequence sample) learnersMean // Train Learners in Paralllel
let mean = Array.minBy averageLoss learnersMean // Chose the learner with best online performance

//Online Learning -- Variance
let errors = PerformanceMeasures.lossesBatch squareLoss sample mean // Obtain losses from the mean predictor
let learnersVariance = Array.map (createSGDLearner positiveRegressionLoss) featureMaps //Setup Candidate
Array.Parallel.iter (learnSequence errors) learnersVariance // Train Learners in Paralllel
let variance = Array.minBy averageLoss learnersVariance // Chose the learner with best online performance

//Charting
let range = [|-float32 Math.PI .. 0.001f .. float32 Math.PI|]

//Plots, notice the usage of a link function in the output of the variances predictions
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