namespace FLearn

///Common loss functions.
[<AutoOpenAttribute>]
module LossFunctions = 
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra
   
    let lossIncurred (loss : LossFunction<'label>) y prediction = loss.Loss y prediction |> fst
    let zeroOneLoss numberOfClasses : LossFunction<int> = 
        {
            Dimension = numberOfClasses - 1
            Loss = fun y prediction ->
                let p = logisticLink prediction
                let predY = Vector.maxIndex p
                if y = predY then Zero, DenseVector.zero (numberOfClasses - 1)
                else One, DenseVector.zero (numberOfClasses - 1)
        }
    let logLoss numberOfClasses : LossFunction<int> =
        let d =  numberOfClasses - 1
        {
            Dimension = d
            Loss = fun y prediction ->
                let Z = prediction |> Vector.fold (fun sum v -> sum + exp(v)) One
                let loss = 
                    if y < d then log(Z) - prediction.[y]
                    else log(Z)
                let Dloss = 
                    DenseVector.init (numberOfClasses - 1)
                        (fun i ->
                            if i = y then exp(prediction.[i]) / Z - One
                            else exp(prediction.[i]) / Z
                        )
                loss, Dloss
            }
    let squareLoss : LossFunction<float32> = 
        {
            Dimension = 1
            Loss = fun y prediction -> 
                pown (y - prediction.[0]) 2, prediction - y
        }
    let quantileLoss q : LossFunction<float32> = 
        {
            Dimension = 1
            Loss = fun y prediction -> 
                if prediction.[0] > y then (One - q) * (prediction.[0] - y), [|One - q|] |> Vector.asVector
                else q * (y - prediction.[0]), [|-q|] |> Vector.asVector
        }
    let positiveRegressionLoss : LossFunction<float32> = 
        { 
            Dimension = 1
            Loss = fun error prediction ->
                0.5f * (exp prediction.[0] + error * exp -prediction.[0]) - sqrt error, 
                0.5f * (exp prediction - error * exp -prediction)
        }