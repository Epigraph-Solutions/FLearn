namespace FLearn

[<AutoOpenAttribute>]
module Learners =
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra

    let predict (predictor: IPredictor<'instance, 'prediction>) (x: 'instance) = predictor.Predict x
    let predictOnline (learner: ILearner<'instance, 'label, 'prediction>) (x: 'instance) = learner.PredictOnline x
    let learn (example: _) (learner: ILearner<'instance, 'label, 'prediction>) = learner.Learn example

    let learnSequence examples (learner: ILearner<'instance, 'label, 'prediction>) =
        Seq.iter (fun s -> learn s learner) examples

    let learnSequenceVerbosely examples (learner: ILearner<'instance, 'label, 'prediction>) =
        let examples =
            examples
            |> Seq.map (fun z ->
                printfn "Count = %A Loss = %A" learner.Count learner.AverageLoss
                z)

        learnSequence examples learner

    let learnDistribution D numberOfSamples (learner: ILearner<'instance, 'label, 'prediction>) =
        let examples = D |> Distribution.samples |> Seq.take numberOfSamples
        learnSequence examples learner

    let learnDistributionVerbosely D numberOfSamples (learner: ILearner<'instance, 'label, 'prediction>) =
        let examples = D |> Distribution.samples |> Seq.take numberOfSamples
        learnSequenceVerbosely examples learner

    let tailAverage (learner: ILearner<'instance, 'label, 'prediction>) = learner.TailAverage()
    let count (learner: ILearner<'instance, 'label, 'prediction>) = learner.Count
    let averageLoss (learner: ILearner<'instance, 'label, 'prediction>) = learner.AverageLoss

    [<StructuredFormatDisplay("count {Count}, loss {AverageLoss}")>]
    type SGDLearner<'instance, 'label, 'prediction> =
        { Features: IFeatureMap<'instance>
          ConvexLossFunction: ConvexLossFunction<'label, 'prediction>
          Weights: Weights
          AverageWeights: Weights
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
            this.Weights * f |> this.ConvexLossFunction.LinkFunction

        member this.Predict(x: 'instance) =
            let f = this.Features.FeatureMap x
            this.AverageWeights * f |> this.ConvexLossFunction.LinkFunction

        member this.Learn(x, y) =
            let loss, gradient =
                let f = this.FeatureMap x
                let prediction = this.Weights * f
                let l, g = this.ConvexLossFunction.ConvexLoss y prediction
                l, Vector.OuterProduct(g, f)

            let newCount = this.Count + One
            let newLearningRate = this.LearningRate + Matrix.normSquared gradient
            let newAverageLoss = this.Count * this.AverageLoss / newCount + loss / newCount

            Matrix.gradientUpdate (this.Weights, gradient, newLearningRate)
            Matrix.runningAverage (this.Count, this.AverageWeights, this.Weights)

            this.Count <- newCount
            this.AverageLoss <- newAverageLoss
            this.LearningRate <- newLearningRate

        member this.TailAverage() =
            this.AverageWeights |> Matrix.mapInPlace (fun _ -> Zero)
            this.AverageLoss <- Zero
            this.Count <- Zero

        interface IPredictor<'instance, 'prediction> with
            member this.Predict x = this.Predict x

        interface ILearner<'instance, 'label, 'prediction> with
            member this.ConvexLossFunction = this.ConvexLossFunction
            member this.Count = this.Count
            member this.AverageLoss = this.AverageLoss
            member this.PredictOnline(x: 'instance) = this.PredictOnline x
            member this.Learn example = this.Learn example
            member this.TailAverage() = this.TailAverage()

    let createSGDLearner loss (features: IFeatureMap<'instance>) =
        let d = features.Dimension

        { Features = features
          ConvexLossFunction = loss
          Count = Zero
          Weights = DenseMatrix.zero loss.Dimension d
          AverageWeights = DenseMatrix.zero loss.Dimension d
          AverageLoss = Zero
          LearningRate = Zero }

    [<StructuredFormatDisplay("count {Count}, loss {AverageLoss}")>]
    type ProjectedSGDLearner<'instance, 'label, 'prediction> =
        { Features: IFeatureMap<'instance>
          ConvexLossFunction: ConvexLossFunction<'label, 'prediction>
          Weights: Weights
          AverageWeights: Weights
          Projection: Projection
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
            this.Weights * f |> this.ConvexLossFunction.LinkFunction

        member this.Predict(x: 'instance) =
            let f = this.Features.FeatureMap x
            this.AverageWeights * f |> this.ConvexLossFunction.LinkFunction

        member this.Learn(x, y) =
            let loss, gradient =
                let f = this.FeatureMap x
                let prediction = this.Weights * f
                let l, g = this.ConvexLossFunction.ConvexLoss y prediction
                l, Vector.OuterProduct(g, f)

            let newCount = this.Count + One
            let newLearningRate = this.LearningRate + Matrix.normSquared gradient
            let newAverageLoss = this.Count * this.AverageLoss / newCount + loss / newCount

            Matrix.gradientUpdate (this.Weights, gradient, newLearningRate)
            this.Projection this.Weights
            Matrix.runningAverage (this.Count, this.AverageWeights, this.Weights)

            this.Count <- newCount
            this.AverageLoss <- newAverageLoss
            this.LearningRate <- newLearningRate

        member this.TailAverage() =
            this.AverageWeights |> Matrix.mapInPlace (fun _ -> Zero)
            this.AverageLoss <- Zero
            this.Count <- Zero

        interface IPredictor<'instance, 'prediction> with
            member this.Predict x = this.Predict x

        interface ILearner<'instance, 'label, 'prediction> with
            member this.ConvexLossFunction = this.ConvexLossFunction
            member this.Count = this.Count
            member this.AverageLoss = this.AverageLoss
            member this.PredictOnline(x: 'instance) = this.PredictOnline x
            member this.Learn example = this.Learn example
            member this.TailAverage() = this.TailAverage()

    let createProjectedSGDLearner loss projection (features: IFeatureMap<'instance>) =
        let d = features.Dimension

        { Features = features
          ConvexLossFunction = loss
          Count = Zero
          Weights = DenseMatrix.zero loss.Dimension d
          AverageWeights = DenseMatrix.zero loss.Dimension d
          Projection = projection
          AverageLoss = Zero
          LearningRate = Zero }

    [<StructuredFormatDisplay("count {Count}, loss {AverageLoss}")>]
    type SGDLearnerNoAverage<'instance, 'label, 'prediction> =
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
            this.Weights * f |> this.ConvexLossFunction.LinkFunction

        member this.Predict(x: 'instance) =
            let f = this.Features.FeatureMap x
            this.Weights * f |> this.ConvexLossFunction.LinkFunction

        member this.Learn(x, y) =
            let loss, gradient =
                let f = this.FeatureMap x
                let prediction = this.Weights * f
                let l, g = this.ConvexLossFunction.ConvexLoss y prediction
                l, Vector.OuterProduct(g, f)

            let newCount = this.Count + One
            let newLearningRate = this.LearningRate + Matrix.normSquared gradient
            let newAverageLoss = this.Count * this.AverageLoss / newCount + loss / newCount

            Matrix.gradientUpdate (this.Weights, gradient, newLearningRate)

            this.Count <- newCount
            this.AverageLoss <- newAverageLoss
            this.LearningRate <- newLearningRate

        member this.TailAverage() =
            this.AverageLoss <- Zero
            this.Count <- Zero

        interface IPredictor<'instance, 'prediction> with
            member this.Predict x = this.Predict x

        interface ILearner<'instance, 'label, 'prediction> with
            member this.ConvexLossFunction = this.ConvexLossFunction
            member this.Count = this.Count
            member this.AverageLoss = this.AverageLoss
            member this.PredictOnline(x: 'instance) = this.PredictOnline x
            member this.Learn example = this.Learn example
            member this.TailAverage() = this.TailAverage()

    let createSGDNoAverageLearner loss (features: IFeatureMap<'instance>) =
        let d = features.Dimension

        { Features = features
          ConvexLossFunction = loss
          Count = Zero
          Weights = DenseMatrix.zero loss.Dimension d
          AverageLoss = Zero
          LearningRate = Zero }

    [<StructuredFormatDisplay("count {Count}, loss {AverageLoss}")>]
    type AdaGradLearner<'instance, 'label, 'prediction> =
        { Features: IFeatureMap<'instance>
          ConvexLossFunction: ConvexLossFunction<'label, 'prediction>
          Weights: Weights
          AverageWeights: Weights
          mutable Count: float32
          mutable AverageLoss: float32
          LearningRate: Weights
          Gsquared: float32 }

        member this.FeatureMap(x: 'instance) = this.Features.FeatureMap x

        member this.Dimension = this.Features.Dimension

        interface IFeatureMap<'instance> with
            member this.FeatureMap(x: 'instance) = this.FeatureMap x
            member this.Dimension = this.Dimension

        member this.PredictOnline(x: 'instance) =
            let f = this.Features.FeatureMap x
            this.Weights * f |> this.ConvexLossFunction.LinkFunction

        member this.Predict(x: 'instance) =
            let f = this.Features.FeatureMap x
            this.AverageWeights * f |> this.ConvexLossFunction.LinkFunction

        member this.Learn(x, y) =
            let loss, gradient =
                let f = this.FeatureMap x
                let prediction = this.Weights * f
                let l, g = this.ConvexLossFunction.ConvexLoss y prediction
                l, Vector.OuterProduct(g, f)

            let newCount = this.Count + One
            let newAverageLoss = this.Count * this.AverageLoss / newCount + loss / newCount

            this.LearningRate
            |> Matrix.mapiInPlace (fun i j eta -> eta + pown (gradient.[i, j]) 2)

            Matrix.adaGradUpdate (this.Weights, gradient, this.LearningRate, this.Gsquared)
            Matrix.runningAverage (this.Count, this.AverageWeights, this.Weights)

            this.Count <- newCount
            this.AverageLoss <- newAverageLoss

        member this.TailAverage() =
            this.AverageWeights |> Matrix.mapInPlace (fun _ -> Zero)
            this.Count <- Zero
            this.AverageLoss <- Zero

        interface IPredictor<'instance, 'prediction> with
            member this.Predict x = this.Predict x

        interface ILearner<'instance, 'label, 'prediction> with
            member this.ConvexLossFunction = this.ConvexLossFunction
            member this.Count = this.Count
            member this.AverageLoss = this.AverageLoss
            member this.PredictOnline(x: 'instance) = this.PredictOnline x
            member this.Learn example = this.Learn example
            member this.TailAverage() = this.TailAverage()

    let createAdaGradLearner g loss (features: IFeatureMap<'instance>) : AdaGradLearner<_, _, _> =
        let d = features.Dimension

        { Features = features
          ConvexLossFunction = loss
          Count = Zero
          Weights = DenseMatrix.zero loss.Dimension d
          AverageWeights = DenseMatrix.zero loss.Dimension d
          AverageLoss = Zero
          Gsquared = g * g
          LearningRate = DenseMatrix.create loss.Dimension d Zero }
