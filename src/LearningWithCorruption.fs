namespace FLearn

module LearningWithCorruption =
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra

    type Reconstruction = Matrix<float32>

    type CorruptedLabel =
        | Clean of MultiClassLabel
        | Corrupted of MultiClassLabel * Reconstruction

    type PartialLoss =
        { Dimension: int //Number of Classes - 1
          Loss: Prediction -> Vector<float32> * Matrix<float32> }

    let internal logLossPartial (numberOfClasses: int) : PartialLoss =
        let d = numberOfClasses - 1

        { Dimension = d
          Loss =
            fun prediction ->
                let Z = prediction |> Vector.fold (fun sum v -> sum + exp (v)) One

                let loss =
                    DenseVector.init numberOfClasses (fun i -> if i < d then log (Z) - prediction.[i] else log (Z))

                let Dloss =
                    DenseMatrix.init numberOfClasses d (fun i j ->
                        if i = j then
                            exp (prediction.[j]) / Z - One
                        else
                            exp (prediction.[j]) / Z)

                loss, Dloss }

    ///Corruption Corrected Losses. Corrects for the Noise in the Labeling Process.
    let CC (L: PartialLoss) : ConvexLossFunction<CorruptedLabel, Prediction> =
        { Dimension = L.Dimension
          ConvexLoss =
            fun y prediction ->
                match y with
                | Clean y -> let loss, Dloss = L.Loss prediction in loss.[y], Dloss.Row(y)
                | Corrupted(y, R) ->
                    let loss, Dloss = L.Loss prediction
                    let reconstructedLoss = R.TransposeThisAndMultiply(loss)
                    let reconstructedDloss = R.TransposeThisAndMultiply(Dloss)
                    reconstructedLoss.[y], reconstructedDloss.Row(y)
          LinkFunction = id
          Loss =
            fun y prediction ->
                match y with
                | Clean y -> let loss, _ = L.Loss prediction in loss.[y]
                | Corrupted(y, R) ->
                    let loss, _ = L.Loss prediction
                    let reconstructedLoss = R.TransposeThisAndMultiply(loss)
                    reconstructedLoss.[y] }

    let logLossCC numberOfClasses = CC(logLossPartial numberOfClasses)
