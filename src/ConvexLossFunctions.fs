namespace FLearn

///Common loss functions.
[<AutoOpenAttribute>]
module ConvexLossFunctions =
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra

    let logLoss numberOfClasses : ConvexLossFunction<int, ProbabilityVector> =
        let d = numberOfClasses - 1

        let convexLoss y prediction =
            let Z = prediction |> Vector.fold (fun sum v -> sum + exp (v)) One
            let loss = if y < d then log (Z) - prediction.[y] else log (Z)

            let Dloss =
                DenseVector.init (numberOfClasses - 1) (fun i ->
                    if i = y then
                        exp (prediction.[i]) / Z - One
                    else
                        exp (prediction.[i]) / Z)

            loss, Dloss

        let linkFunction (prediction: Prediction) =
            let d = prediction.Count
            let Z = prediction |> Vector.fold (fun sum v -> sum + exp (v)) One

            let result =
                DenseVector.init (d + 1) (fun i -> if i < d then exp (prediction.[i]) / Z else One / Z)

            result

        let loss (label: int) (prediction: ProbabilityVector) = - log(prediction.[label])

        { Dimension = d
          ConvexLoss = convexLoss
          LinkFunction = linkFunction
          Loss = loss }

    let squareLoss: ConvexLossFunction<float32, float32> =
        { Dimension = 1
          ConvexLoss = fun y prediction -> pown (y - prediction.[0]) 2, prediction - y
          LinkFunction = fun prediction -> prediction.[0]
          Loss = fun y prediction -> pown (y - prediction) 2 }

    let quantileLoss q : ConvexLossFunction<float32, float32> =
        { Dimension = 1
          ConvexLoss =
            fun y prediction ->
                if prediction.[0] > y then
                    (One - q) * (prediction.[0] - y), [| One - q |] |> Vector.asVector
                else
                    q * (y - prediction.[0]), [| -q |] |> Vector.asVector
          LinkFunction = fun prediction -> prediction.[0]
          Loss =
            fun y prediction ->
                if prediction > y then
                    (One - q) * (prediction - y)
                else
                    q * (y - prediction) }

    let positiveRegressionLoss: ConvexLossFunction<float32, float32> =
        { Dimension = 1
          ConvexLoss =
            fun error prediction ->
                0.5f * (exp prediction.[0] + error * exp -prediction.[0]) - sqrt error,
                0.5f * (exp prediction - error * exp -prediction)
          LinkFunction = fun prediction -> exp (2.0f * prediction.[0])
          Loss = fun error prediction -> 0.5f * (sqrt prediction + error / sqrt prediction) - sqrt error }
