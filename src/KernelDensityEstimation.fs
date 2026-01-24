namespace FLearn

module GaussianMeanMethods =
    open FLearn.Utilities
    open FLearn.Covers
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.Distributions

    type GaussianKernelDensityEstimator(sigma: float32, instances: seq<Vector<float32>>, weights: Vector<float32>) =
        let instanceMatrix = DenseMatrix.ofRowSeq instances
        let norms2 = instances |> Seq.map (fun v -> v * v) |> Seq.toArray
        let g sigma x = exp (-x / (Two * pown sigma 2))

        member this.Dimension = instanceMatrix.RowCount
        member this.Instances = instanceMatrix |> Matrix.toRowSeq
        member this.Weights = weights

        member this.Sigma = sigma

        member this.FeatureMap(x': Vector<float32>) =
            let norm2x = x' * x'

            instanceMatrix * x'
            |>! Vector.mapiInPlace (fun i ip_rep_x -> g sigma (norms2.[i] + norm2x - Two * ip_rep_x))

        interface IFeatureMap<Vector<float32>> with
            member this.FeatureMap(x: Vector<float32>) = this.FeatureMap x
            member this.Dimension = this.Dimension

        member this.Predict(x) = this.Weights * this.FeatureMap x

        interface IPredictor<Vector<float32>, float32> with
            member this.Predict x = this.Predict x

        member this.KernelMatrix() =
            instanceMatrix.TransposeAndMultiply(instanceMatrix)
            |>! Matrix.mapiInPlace (fun i j ip -> g sigma (norms2.[i] + norms2.[j] - Two * ip))

        member this.NormSquared() =
            let kernelMatrix = this.KernelMatrix()
            (weights * kernelMatrix * weights)

        member this.Projection(sample: Vector<float32> seq) =
            let K =
                GaussianKernelDensityEstimator(this.Sigma, sample, DenseVector.raw [| 0.0f |]).KernelMatrix()

            let Z = sample |> Seq.map (fun x -> this.Predict x) |> DenseVector.ofSeq
            K.SolveIterative(Z, MathNet.Numerics.LinearAlgebra.Single.Solvers.BiCgStab())

        member this.Reduce(maxIterations, error) =
            let cover = gaussianKernelCover maxIterations error this.Sigma this.Instances
            let w = this.Projection(cover)
            GaussianKernelDensityEstimator(this.Sigma, cover, w)


        member this.ReduceParallel(chunkSize, maxIterations, error) =
            let cover =
                gaussianKernelCoverParallel chunkSize maxIterations error this.Sigma this.Instances

            let w = this.Projection(cover)
            GaussianKernelDensityEstimator(this.Sigma, cover, w)

    let createGaussianKernelDensityEstimator sigma sample weights =
        GaussianKernelDensityEstimator(sigma, sample, weights)

    let equalWeights sigma (sample: Vector<float32> seq) =
        let n = Seq.length sample
        GaussianKernelDensityEstimator(sigma, sample, DenseVector.create n (1.0f / float32 n))

    let reduce maxIterations error (G: GaussianKernelDensityEstimator) = G.Reduce(maxIterations, error)

    let reduceParallel chunkSize maxIterations error (G: GaussianKernelDensityEstimator) =
        G.ReduceParallel(chunkSize, maxIterations, error)
