namespace FLearn

module GaussianMeanMethods =
    open FLearn.Utilities
    open FLearn.Covers
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.Distributions

    type GaussianKernelDensityEstimator(sigma: float32, instances: seq<Vector<float32>>) =
        let instanceMatrix = DenseMatrix.ofRowSeq instances
        let norms2 = instances |> Seq.map (fun v -> v * v) |> Seq.toArray
        let g sigma x = exp (-x / (Two * pown sigma 2))

        member this.Dimension = instanceMatrix.RowCount
        member this.Instances = instanceMatrix |> Matrix.toRowSeq

        static member Collect(herds: seq<GaussianKernelDensityEstimator>) =
            let instances = herds |> Seq.collect (fun h -> h.Instances)

            let sigma = Seq.head herds |> fun x -> x.Sigma
            GaussianKernelDensityEstimator(sigma, instances)

        member this.Sigma = sigma

        member this.FeatureMap(x': Vector<float32>) =
            let norm2x = x' * x'

            instanceMatrix * x'
            |>! Vector.mapiInPlace (fun i ip_rep_x -> g sigma (norms2.[i] + norm2x - Two * ip_rep_x))

        interface IFeatureMap<Vector<float32>> with
            member this.FeatureMap(x: Vector<float32>) = this.FeatureMap x
            member this.Dimension = this.Dimension

        member this.KernelMatrix() =
            instanceMatrix.TransposeAndMultiply(instanceMatrix)
            |>! Matrix.mapiInPlace (fun i j ip -> g sigma (norms2.[i] + norms2.[j] - Two * ip))

        member this.NormSquared(weights: Vector<float32>) =
            let kernelMatrix = this.KernelMatrix()
            (weights * kernelMatrix * weights)

        member this.Cover(maxIterations, error) : GaussianKernelDensityEstimator =
            gaussianKernelCover maxIterations error this.Sigma this.Instances
            |> fun x -> GaussianKernelDensityEstimator(this.Sigma, x)

    let gaussianKernelDensityEstimator sigma instances =
        GaussianKernelDensityEstimator(sigma, instances)

    module GaussianTools =

        let equalWeights (h: GaussianKernelDensityEstimator) =
            let n = h.Dimension
            DenseVector.create n (1.0f / float32 n), h

        let collectGaussianKernelDensityEstimators (herds: seq<GaussianKernelDensityEstimator>) =
            GaussianKernelDensityEstimator.Collect herds

        let cover error (h: GaussianKernelDensityEstimator) = h.Cover(error)

        let projection (f2: GaussianKernelDensityEstimator) (w: Vector<float32>, f1: GaussianKernelDensityEstimator) =
            if f1.Sigma = f2.Sigma then
                let K = f2.KernelMatrix()
                let Z = f2.Instances |> Seq.map (fun x -> w * f1.FeatureMap(x)) |> DenseVector.ofSeq
                K.SolveIterative(Z, MathNet.Numerics.LinearAlgebra.Single.Solvers.BiCgStab())
            else
                failwith "Sigma must be the same for both f1 and f2."

        let reduce error (w, f) =
            let c = cover error f
            let w = projection c (w, f)
            w, c
