#r "nuget: FLearn"
#r "nuget: FSharpPlus"
#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: XPlot.Plotly"

open System
open FLearn
open FSharpPlus
open XPlot.Plotly
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions

//Helpers
let toVector (x, y) : Vector<float32> = DenseVector.raw [|x ; y|]
let plot (sample : seq<_ * _>) = 
    let scatter = Scatter(x = Seq.map fst sample, y = Seq.map snd sample, mode = "markers")
    Chart.Plot(scatter).Show()

let rnd = Distribution.rnd
let normal mean sigma = 
    Distribution.create (fun () -> Sample.normal mean sigma rnd)
let uniform lower upper = 
    Distribution.create (fun () -> Sample.continuousUniform lower upper rnd)

//Distributions
let P sigma = 
    distribution {
        let! x = uniform -Math.PI Math.PI
        let! y = normal x sigma
        return float32 x, float32 y
    }

let indepenceRating (P : Distribution<float32 * float32>) = 
    let N = (uncurry zip) (unzip P)
    let D = 
        distribution {
            let! label = Distribution.coinFlip 0.5f
            if label then 
                let! x = P
                return toVector x, 1
            else
                let! x = N
                return toVector x, 0
        }

    let loss = logLoss 2
    let features = randomFourierFeatureMap 1.0f 2 500
    let learner = createSGDLearner loss features

    learnDistribution D 10000 learner
    averageLoss learner

let independences = [|0.1 .. 0.1 .. 2.0|] |>> (P >> indepenceRating)

Scatter(x = [0 .. independences.Length - 1], y = independences) |> Chart.Plot |> fun x -> x.Show()