(**
One strength of Gaussian Processes is that they provide both an estimate of the conditional mean, and the conditional variance.
Here we show how one can achieve this using a model that is trained on the errors of model trained to find the conditional mean.
Along the way we utilize a sparse approximation of the Gaussian kernel feature map.

One issue with doing regression on errors, is that errors are always positive! We bake this requirement into the loss used for doing the error 
estimation.
*)

#r "nuget: MathNet.Numerics"
#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: MathNet.Numerics.MKL.Win"
#r "nuget: FLearn"
#r "nuget: XPlot.Plotly"

open System
open FLearn
open XPlot.Plotly
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions

MathNet.Numerics.Control.UseNativeMKL()

//Helpers -- Just some distributions and conversions between vectors and floats.
let rnd = Distribution.rnd

let normal mean sigma =
    Distribution.create (fun () -> Sample.normal mean sigma rnd)

let uniform lower upper =
    Distribution.create (fun () -> Sample.continuousUniform lower upper rnd)

let toVector (x: float32) = DenseVector.raw [| x |]

//Target Distribution, note the non uniformity of the error.
let Px = uniform -Math.PI Math.PI

let trueMean x =
    if x < 0.0 then 2.0 * x else sin (2.0 * x)

let noise x = 0.75 * cos (2.0 * x) |> abs

let P =
    distribution {
        let! x = Px
        let! y = normal (trueMean x) (noise x)
        return float32 x, float32 y
    }

//Sample form Target Distribution.
let n = 20000
let sample = P |> Distribution.samples |> Seq.take n |> Seq.toArray

//Setting up Candidate Feature Maps.
let features sigma =
    sample
    |> Array.map (fst >> toVector) // Extract x from the sample
    |> Covers.gaussianKernelCover 50 0.0001f sigma
    |> nystromFeatureMap sigma
    |> preProcess toVector

let featureMaps = Array.Parallel.map features [| 0.1f..0.01f..0.4f |] //Setup the Feature Maps.

//Online Learning -- Mean.
let learnersMean = Array.map (createSGDLearner squareLoss) featureMaps // Setup candidate learners for the conditional mean.
Array.Parallel.iter (learnSequence sample) learnersMean // Train learners in Parallel.
let mean = Array.minBy averageLoss learnersMean // Chose the learner with best online performance.

//Online Learning -- Variance.
let errors = PerformanceMeasures.lossesBatch squareLoss sample mean // Obtain losses from the mean predictor.

let learnersVariance =
    Array.map (createSGDLearner positiveRegressionLoss) featureMaps // Setup Candidate learners for the conditional variance.

Array.Parallel.iter (learnSequence errors) learnersVariance // Train Learners in Parallel.
let variance = Array.minBy averageLoss learnersVariance // Chose the learner with best online performance.

//Charting
let range = [| -float32 Math.PI .. 0.001f .. float32 Math.PI |]

//Plots, notice the usage of a link function in the output of the variances predictions.
[ Scatter(x = Array.map fst sample, y = Array.map snd sample, mode = "markers", name = "Data")
  Scatter(x = range, y = Array.map mean.Predict range, marker = Marker(color = "DarkRed"), name = "Mean")
  Scatter(
      x = range,
      y = Array.map (fun x -> mean.Predict x + 3.0f * sqrt (variance.Predict x)) range,
      marker = Marker(color = "Red"),
      name = "Upper Bound"
  )
  Scatter(
      x = range,
      y = Array.map (fun x -> mean.Predict x - 3.0f * sqrt (variance.Predict x)) range,
      marker = Marker(color = "Red"),
      name = "Lower Bound"
  ) ]
|> Chart.Plot
|> Chart.Show
