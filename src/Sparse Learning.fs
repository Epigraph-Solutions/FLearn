namespace FLearn

module SparseLearning =
    open FLearn.Utilities
    open FLearn.ConvexLossFunctions
    open MathNet.Numerics.LinearAlgebra

    [<StructuredFormatDisplay("count {Count}, loss {AverageLoss}")>]
    type SparseSGDNoAverageLearner<'instance, 'label, 'prediction> =
        { Features: IFeatureMap<'instance>
          ConvexLossFunction: ConvexLossFunction<'label, 'prediction>
          Weights: Weights
          mutable Count: float32
          mutable AverageLoss: float32
          mutable LearningRate: float32 }

        member this.FeatureMap(x: 'instance) = this.Features.FeatureMap x

        member this.Dimension = this.Features.Dimension

        interface IFeatureMap<'instance> with
            member this.FeatureMap(x: 'instance) = this.FeatureMap x
            member this.Dimension = this.Dimension

        member this.PredictOnline(x: 'instance) =
            let f = this.Features.FeatureMap x
            Matrix.sparseMultiply (this.Weights, f) |> this.ConvexLossFunction.LinkFunction

        member this.Learn(x, y) =
            let loss, gradient =
                let f = this.FeatureMap x
                let prediction = Matrix.sparseMultiply (this.Weights, f)
                let l, g = this.ConvexLossFunction.ConvexLoss y prediction
                l, Vector.sparseOuterProduct (g, f)

            let newCount = this.Count + One
            let newLearningRate = this.LearningRate + Matrix.sparseNormSquared gradient
            let newAverageLoss = this.Count * this.AverageLoss / newCount + loss / newCount

            Matrix.sparseGradientUpdate (this.Weights, gradient, newLearningRate)

            this.Count <- newCount
            this.AverageLoss <- newAverageLoss
            this.LearningRate <- newLearningRate

        member this.TailAverage() =
            this.AverageLoss <- Zero
            this.Count <- Zero

        interface IPredictor<'instance, 'prediction> with
            member this.Predict x = this.PredictOnline x

        interface ILearner<'instance, 'label, 'prediction> with
            member this.ConvexLossFunction = this.ConvexLossFunction
            member this.Count = this.Count
            member this.AverageLoss = this.AverageLoss
            member this.Predict x = this.PredictOnline x
            member this.PredictOnline x = this.PredictOnline x
            member this.Learn example = this.Learn example
            member this.TailAverage() = this.TailAverage()

    let createSparseSGDNoAverageLearner loss (features: IFeatureMap<'instance>) =
        let d = features.Dimension

        { Features = features
          ConvexLossFunction = loss
          Count = Zero
          Weights = SparseMatrix.zero loss.Dimension d
          AverageLoss = Zero
          LearningRate = Zero }
