namespace FLearn

[<AutoOpenAttribute>]
module FeatureMaps =
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.Distributions

    let featureMap (F: #IFeatureMap<'instance>) (x: 'instance) = F.FeatureMap x
    let dimension (F: #IFeatureMap<'instance>) = F.Dimension

    ///Adds preprocessing step given by the function f.
    let preProcess (f: 'rawData -> 'instance) (F: #IFeatureMap<'instance>) =
        { new IFeatureMap<'rawData> with
            member this.FeatureMap(x: 'rawData) = F.FeatureMap(f x)
            member this.Dimension = F.Dimension }

    ///Composes feature maps yield F1 >> F2.
    let postProcess (F2: #IFeatureMap<Vector<float32>>) (F1: #IFeatureMap<'instance>) =
        { new IFeatureMap<'instance> with
            member this.FeatureMap(x: _) = F2.FeatureMap(F1.FeatureMap x)
            member this.Dimension = F2.Dimension }
    //Forms the direct sum of F1 and F2.
    let addFeatures (F1: #IFeatureMap<'instance>) (F2: #IFeatureMap<'instance>) =
        { new IFeatureMap<'instance> with
            member this.Dimension = F1.Dimension + F2.Dimension

            member this.FeatureMap x =
                let v1 = F1.FeatureMap x
                let v2 = F2.FeatureMap x

                seq {
                    yield! v1
                    yield! v2
                }
                |> DenseVector.ofSeq }

    ///Forms the direct sum of all features.
    let sumFeatures (features: seq<#IFeatureMap<'instance>>) =
        { new IFeatureMap<'instance> with
            member this.Dimension = features |> Seq.sumBy (fun x -> x.Dimension)

            member this.FeatureMap x =
                seq {
                    for f in features do
                        yield! f.FeatureMap x
                }
                |> DenseVector.ofSeq }

    ///Forms the cross product of features with itself.
    let squareFeatures (F: #IFeatureMap<'x>) =
        { new IFeatureMap<'x> with
            member this.Dimension = F.Dimension * F.Dimension

            member this.FeatureMap x =
                let v = F.FeatureMap x

                seq {
                    for v1 in v do
                        for v2 in v do
                            yield v1 * v2
                }
                |> DenseVector.ofSeq }

    ///Forms the cross product of F1 and F2.
    let crossFeatures (F1: #IFeatureMap<'instance>) (F2: #IFeatureMap<'instance>) =
        { new IFeatureMap<'instance> with
            member this.Dimension = F1.Dimension * F2.Dimension

            member this.FeatureMap x =
                let v1 = F1.FeatureMap x
                let v2 = F2.FeatureMap x

                seq {
                    for v1 in v1 do
                        for v2 in v2 do
                            yield v1 * v2
                }
                |> DenseVector.ofSeq }

    ///Standard linear feature map.
    let identityFeatureMap (dimension: int) =
        { new IFeatureMap<Vector<float32>> with
            member this.Dimension = dimension
            member this.FeatureMap(x: Vector<float32>) = x }

    ///Randomized approximation to polynomial feature map of order k.
    let randomPolynomialMap k inputDimension nFeatures =
        let Z = One / sqrt (float32 nFeatures)

        let W =
            let randomMatrix () =
                DenseMatrix.init nFeatures inputDimension (fun _ _ ->
                    if Distribution.rnd.Next(2) = 0 then 1.0f else -1.0f)

            Array.init k (fun _ -> randomMatrix ())

        { new IFeatureMap<Vector<float32>> with
            member __.FeatureMap x =
                let V = Array.map (fun w -> w * x) W

                Array.Parallel.init nFeatures (fun i -> Array.fold (fun state (v: Vector<_>) -> state * v.[i]) Z V)
                |> DenseVector.raw

            member __.Dimension = nFeatures }

    ///Randomized approximation to the Gaussian kernel feature map.
    let randomFourierFeatureMap (sigma: float32) inputDimension numberOfFeatures =
        let sigma = float sigma
        let normalization = sqrt (2.0f / float32 numberOfFeatures)

        let projectionMatrix =
            DenseMatrix.init numberOfFeatures inputDimension (fun _ _ ->
                Sample.normal 0.0 (1.0 / sigma) Distribution.rnd |> float32)

        let offSets =
            DenseVector.init numberOfFeatures (fun _ ->
                Sample.continuousUniform 0.0 (2.0 * System.Math.PI) Distribution.rnd |> float32)

        { new IFeatureMap<_> with
            member __.FeatureMap(x: Vector<float32>) =
                projectionMatrix * x
                |>! Vector.mapiInPlace (fun i x -> normalization * cos (x + offSets.[i]))

            member __.Dimension = numberOfFeatures }

    let nystromFeatureMap (sigma: float32) (examples: Vector<float32> seq) =
        let d = Seq.length examples
        let instanceMatrix = DenseMatrix.ofRowSeq examples
        let norms2 = examples |> Seq.map (fun v -> v * v) |> Seq.toArray
        let g sigma x = exp (-x / (Two * pown sigma 2))

        let K =
            instanceMatrix.TransposeAndMultiply(instanceMatrix)
            |>! Matrix.mapiInPlace (fun i j ip -> g sigma (norms2.[i] + norms2.[j] - Two * ip))

        let svd = K.Svd()
        let U, S, VT = svd.U, svd.S, svd.VT
        let D = S |> Vector.map (fun s -> One / sqrt (s)) |> DenseMatrix.ofDiag
        let inverseK = U * D * VT

        { new IFeatureMap<Vector<float32>> with
            member this.Dimension = d

            member this.FeatureMap x =
                let norm2x = x * x
                let M = instanceMatrix * x

                M
                |> Vector.mapiInPlace (fun i ip_rep_x -> g sigma (norms2.[i] + norm2x - Two * ip_rep_x))

                inverseK * M }
