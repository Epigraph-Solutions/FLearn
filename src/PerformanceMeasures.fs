namespace FLearn

module PerformanceMeasures = 
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra

    let lossBatch (loss : LossFunction<'label>) testSet (predictor : IPredictor<'instance>) = 
        testSet
        |> Seq.fold (fun (totalLoss, n) (x, y) ->
            let p = predictor.Predict x
            let l = loss.Loss y p |> fst
            totalLoss + l, n + One
        ) (Zero, Zero)
        |> fun (loss, n) -> loss / n
    let lossesBatch (loss : LossFunction<'label>) testSet (predictor : IPredictor<'instance>) =
        testSet
        |> Seq.map (fun (x, y) ->
            let p = predictor.Predict x
            x, loss.Loss y p |> fst
        )
    let lossSummary (losses : LossFunction<'label> []) testSet (predictor : IPredictor<'instance>) =
        let loss = DenseVector.zero losses.Length
        let n = 
            testSet
            |> Seq.fold (fun n (x, y) ->
                let p = predictor.Predict x
                loss 
                |> Vector.mapiInPlace (fun i l -> 
                    l + fst (losses.[i].Loss y p) 
                )
                n + One
            ) Zero
        loss / n |> Vector.asArray

    ///Summarizes false positive, false negative, true positive and true negative rates of a classifier. A Prediction > 0.0 corresponds to a Predicted positive label.
    type ConfusionMatrix = 
        {
            TruePositive : float32
            FalsePositive : float32
            TrueNegative : float32
            FalseNegative : float32
            Bias : float32
        }
        member this.Count = 
            this.TruePositive + this.FalsePositive + this.TrueNegative + this.FalseNegative
        member this.PostitiveCount = 
            this.TruePositive + this.FalseNegative
        member this.NegativeCount = 
            this.TrueNegative + this.FalsePositive
    ///Sum of true positive and true negative rates.
    let accuracy (C : ConfusionMatrix) = 
        (C.TruePositive + C.TrueNegative) / C.Count
    ///True positive rate.
    let truePositiveRate (C : ConfusionMatrix) = 
        C.TruePositive / C.PostitiveCount
    ///True negative rate.
    let trueNegativeRate (C : ConfusionMatrix) = 
        C.TrueNegative / C.NegativeCount
    ///Average of true positive and true negative rates.
    let balancedAccuracy (C : ConfusionMatrix) = 
        OneHalf * (truePositiveRate C + trueNegativeRate C)
    ///Percentage of actual positives in PredictOnlineed positives.
    let positivePrecision (C : ConfusionMatrix) = 
        C.TruePositive / (C.TruePositive + C.FalsePositive)
    ///Percentage of actual negatives in PredictOnlineed negatives.
    let negativePrecision (C : ConfusionMatrix) = 
        C.TrueNegative / (C.TrueNegative + C.FalseNegative)

    let rocCurve confusionCurve =
        confusionCurve
        |> Array.map (fun c -> truePositiveRate c, trueNegativeRate c, c.Bias)
    let AUC (rocCurve : (float32 * float32 * _) []) = 
        rocCurve
        |> Seq.pairwise
        |> Seq.fold (fun sum ((x1, y1, _), (x2, y2, _)) ->
            let width = x2 - x1
            sum + width * y1 
        ) Zero

    type ConfustionCurve = 
        {
            ConfusionCurve : ConfusionMatrix []
            RocCurve : (float32 * float32 * float32) []
            AUC : float32
        }
    let createConfusionCurve (testSet : ('instance * int) []) (predictor : IPredictor<'instance>) =
        let scoredLabels = testSet |> Seq.map (fun (x, y) -> let p = predictor.Predict x in y, p)
        let sortedScoredLabels = scoredLabels |> Seq.sortBy (fun (y, p) -> p.[0])
        let nNegatives, nPositives, n = 
            sortedScoredLabels
            |> Seq.fold (fun (neg, pos, n) (y,_) ->
                match y with
                | 1 -> neg + One, pos, n + One
                | 0 -> neg, pos + One, n + One
                | _ -> failwith "Unexpected Label"
            ) (Zero, Zero, Zero)
        let initialConfusionMatrix = 
            {
                TruePositive = nPositives
                FalsePositive = nNegatives
                TrueNegative = Zero
                FalseNegative = Zero
                Bias = (snd (Seq.head sortedScoredLabels)).[0]
            }
        
        let confusionList = 
            sortedScoredLabels 
            |> Seq.fold (fun cl (y, p) -> 
                let C = List.head cl
                let newC = 
                    match y with
                    | 0 ->
                        {C with TruePositive = C.TruePositive - One ; FalseNegative = C.FalseNegative + One ; Bias = p.[0]}
                    | 1 ->
                        {C with TrueNegative = C.TrueNegative + One ; FalsePositive = C.FalsePositive - One ; Bias = p.[0]}
                    | _ -> failwith "Unexpected Label"
                newC :: cl
            ) [initialConfusionMatrix]

        let confusionCurve = List.toArray confusionList
        let rocCurve = rocCurve confusionCurve
        let auc = AUC rocCurve
        {
            ConfusionCurve = confusionCurve
            RocCurve = rocCurve
            AUC = auc
        }