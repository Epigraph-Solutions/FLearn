namespace FLearn

///Common loss functions.
[<AutoOpenAttribute>]
module ConvexLossFunctions =
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra

    let canonicalBrier: ConvexLossFunction<BinaryLabel, Score> =
        { Dimension = 1
          ConvexLoss =
            fun y prediction ->
                let alpha = prediction.[0]

                let psi =
                    if alpha < -2.0f then -alpha
                    elif alpha > 2.0f then alpha
                    else One + 0.25f * alpha * alpha

                let dspi =
                    if alpha < -2.0f then -One
                    elif alpha > 2.0f then One
                    else 0.5f * alpha

                match y with
                | Positive -> -alpha + psi, -One + Vector.asVector [| dspi |]
                | Negative -> alpha + psi, One + Vector.asVector [| dspi |]

          LinkFunction = fun prediction -> prediction.[0]
          Loss =
            fun y prediction ->
                let alpha = prediction

                let psi =
                    if alpha < -2.0f then -alpha
                    elif alpha > 2.0f then alpha
                    else One + 0.25f * alpha * alpha

                match y with
                | Positive -> -alpha + psi
                | Negative -> alpha + psi }

    let canonicalHinge: ConvexLossFunction<BinaryLabel, BinaryLabel> =
        { Dimension = 1
          ConvexLoss =
            fun y prediction ->
                let alpha = prediction.[0]

                let psi =
                    if alpha < -One then -OneHalf * alpha
                    elif alpha > One then OneHalf * alpha
                    else OneHalf

                let dspi =
                    if alpha < -One then -OneHalf
                    elif alpha > One then OneHalf
                    else Zero

                match y with
                | Positive -> -OneHalf * alpha + psi, Vector.asVector [| -OneHalf + dspi |]
                | Negative -> OneHalf * alpha + psi, Vector.asVector [| OneHalf + dspi |]

          LinkFunction =
            fun prediction ->
                let score = prediction.[0]
                if score > Zero then Positive else Negative

          Loss = fun y prediction -> if y = prediction then Zero else One }

    let canonicalCSHinge c : ConvexLossFunction<BinaryLabel, BinaryLabel> =
        { Dimension = 1
          ConvexLoss =
            fun y prediction ->
                let alpha = prediction.[0]

                let psi =
                    if alpha < -One then -OneHalf * alpha
                    elif alpha > c then OneHalf * alpha
                    else (c - OneHalf * alpha + OneHalf * c * alpha) / (One + c)

                let dspi =
                    if alpha < -One then -OneHalf
                    elif alpha > c then OneHalf
                    else OneHalf * (c - One) / (One + c)

                match y with
                | Positive -> -OneHalf * alpha + psi, Vector.asVector [| -OneHalf + dspi |]
                | Negative -> OneHalf * alpha + psi, Vector.asVector [| OneHalf + dspi |]

          LinkFunction =
            fun prediction ->
                let score = prediction.[0]
                if score > (c - 1.0f) / 2.0f then Positive else Negative
          Loss =
            fun y prediction ->
                match y, prediction with
                | Positive, Negative -> One
                | Negative, Positive -> c
                | _ -> Zero }

    let logLoss numberOfClasses : ConvexLossFunction<MultiClassLabel, ProbabilityVector> =
        let d = numberOfClasses - 1

        let convexLoss y prediction =
            let Z = prediction |> Vector.fold (fun sum v -> sum + exp v) One
            let loss = if y < d then log Z - prediction.[y] else log Z

            let Dloss =
                DenseVector.init (numberOfClasses - 1) (fun i ->
                    if i = y then
                        exp prediction.[i] / Z - One
                    else
                        exp prediction.[i] / Z)

            loss, Dloss

        let linkFunction (prediction: Prediction) =
            let d = prediction.Count
            let Z = prediction |> Vector.fold (fun sum v -> sum + exp v) One

            let result =
                DenseVector.init (d + 1) (fun i -> if i < d then exp prediction.[i] / Z else One / Z)

            result

        let loss (label: int) (prediction: ProbabilityVector) = -log(prediction.[label])

        { Dimension = d
          ConvexLoss = convexLoss
          LinkFunction = linkFunction
          Loss = loss }

    let squareLoss: ConvexLossFunction<Real, Real> =
        { Dimension = 1
          ConvexLoss = fun y prediction -> pown (y - prediction.[0]) 2, prediction - y
          LinkFunction = fun prediction -> prediction.[0]
          Loss = fun y prediction -> pown (y - prediction) 2 }

    let quantileLoss q : ConvexLossFunction<Real, Real> =
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

    let positiveRegressionLoss: ConvexLossFunction<Real, Real> =
        { Dimension = 1
          ConvexLoss =
            fun error prediction ->
                0.5f * (exp prediction.[0] + error * exp -prediction.[0]) - sqrt error,
                0.5f * (exp prediction - error * exp -prediction)
          LinkFunction = fun prediction -> exp (2.0f * prediction.[0])
          Loss = fun error prediction -> 0.5f * (sqrt prediction + error / sqrt prediction) - sqrt error }
