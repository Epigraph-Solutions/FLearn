namespace FLearn

module LearningWithCorruption =
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra

    type Reconstruction = Matrix<float32>

    type CorruptedLabel =
        | Clean of int
        | Corrupted of int * Reconstruction

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

    let internal zeroOneLossPartial (numberOfClasses: int) : PartialLoss =
        let d = numberOfClasses - 1

        { Dimension = numberOfClasses - 1
          Loss =
            fun prediction ->
                let p = logisticLink prediction
                let predY = Vector.maxIndex p

                let loss =
                    DenseVector.init numberOfClasses (fun i -> if i = predY then Zero else One)

                let Dloss = SparseMatrix.zero numberOfClasses d
                loss, Dloss }

    ///Corruption Corrected Losses. Corrects for the Noise in the Labeling Process.
    let CC (L: PartialLoss) : LossFunction<CorruptedLabel> =
        { Dimension = L.Dimension
          Loss =
            fun y prediction ->
                match y with
                | Clean y -> let loss, Dloss = L.Loss prediction in loss.[y], Dloss.Row(y)
                | Corrupted(y, R) ->
                    let loss, Dloss = L.Loss prediction
                    let reconstructedLoss = R.TransposeThisAndMultiply(loss)
                    let reconstructedDloss = R.TransposeThisAndMultiply(Dloss)
                    reconstructedLoss.[y], reconstructedDloss.Row(y) }

    let logLossCC numberOfClasses = CC(logLossPartial numberOfClasses)
    let zeroOneLossCC numberOfClasses = CC(zeroOneLossPartial numberOfClasses)
