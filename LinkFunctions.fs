namespace FLearn

///Common link functions.
[<AutoOpenAttribute>]
module LinkFunctions = 
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra

    let standardLink (prediction : Prediction) = prediction.[0]
    let positiveRegressionLink (prediction : Prediction) = exp(2.0f * prediction.[0])
    let logisticLink (prediction : Prediction) = 
        let d = prediction.Count
        let Z = prediction |> Vector.fold (fun sum v -> sum + exp(v)) One
        DenseVector.init (d + 1) (fun i -> if i < d  then exp(prediction.[i]) / Z else One / Z)
