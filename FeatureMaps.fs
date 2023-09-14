namespace FLearn

[<AutoOpenAttribute>]
module FeatureMaps = 
    open FLearn.Utilities
    open FLearn.Herding
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.Distributions

    let featureMap (F : #IFeatureMap<'instance>) (x : 'instance) = F.FeatureMap x
    let dimension (F : #IFeatureMap<'instance>) = F.Dimension

    ///Adds preproccessing step given by the function f.
    let preProcess (f : 'rawData -> 'instance) (F : #IFeatureMap<'instance>) = 
        {
            new IFeatureMap<'rawData> with
                member this.FeatureMap (x : 'rawData) = F.FeatureMap (f x)
                member this.Dimension = F.Dimension
        }
    ///Composes feature maps yield F1 >> F2.
    let postProcess (F2 : #IFeatureMap<Vector<float32>>) (F1 : #IFeatureMap<'instance>) =
        {
            new IFeatureMap<'instance> with
                member this.FeatureMap (x : _) = F2.FeatureMap (F1.FeatureMap x)
                member this.Dimension = F2.Dimension
        }
    //Forms the direct sum of F1 and F2.
    let addFeatures (F1 : #IFeatureMap<'instance>) (F2 : #IFeatureMap<'instance>) = 
        {
            new IFeatureMap<'instance> with
                member this.Dimension = F1.Dimension + F2.Dimension
                member this.FeatureMap x = 
                    let v1 = F1.FeatureMap x 
                    let v2 = F2.FeatureMap x
                    seq{
                        yield! v1
                        yield! v2
                    }
                    |> DenseVector.ofSeq

        }
    ///Froms the direct sum of all features.
    let sumFeatures (features : seq<#IFeatureMap<'instance>>) = 
        {
            new IFeatureMap<'instance> with
                member this.Dimension = features |> Seq.sumBy (fun x -> x.Dimension)
                member this.FeatureMap x = 
                    seq{
                        for f in features do 
                            yield! f.FeatureMap x
                    }
                    |> DenseVector.ofSeq

        }
    ///Forms the cross product of features with itself.
    let squareFeatures (F : #IFeatureMap<'x>) = 
        {
            new IFeatureMap<'x> with
                member this.Dimension = F.Dimension * F.Dimension
                member this.FeatureMap x = 
                    let v = F.FeatureMap x
                    seq{
                        for v1 in v do
                            for v2 in v do
                                yield v1 * v2
                    }
                    |> DenseVector.ofSeq
        }
    ///Forms the cross product of F1 and F2.
    let crossFeatures (F1 : #IFeatureMap<'instance>) (F2 : #IFeatureMap<'instance>) = 
        {
            new IFeatureMap<'instance> with
                member this.Dimension = F1.Dimension * F2.Dimension
                member this.FeatureMap x =
                    let v1 = F1.FeatureMap x
                    let v2 = F2.FeatureMap x
                    seq{
                        for v1 in v1 do
                            for v2 in v2 do
                                yield v1 * v2
                    }
                    |> DenseVector.ofSeq
        }

    ///Standard linear feature map.
    let identityFeatureMap (dimension : int) = 
        {
            new IFeatureMap<Vector<float32>> with
                member this.Dimension = dimension
                member this.FeatureMap (x : Vector<float32>) = x
        }

    ///Randomized approximation to polynomial feature map of order k.
    let randomPolynomialMap k inputDimension nFeatures = 
        let Z = One / sqrt(float32 nFeatures)
        let W = 
            let randomGaussianMatrix () = 
                DenseMatrix.init 
                    nFeatures 
                    inputDimension 
                    (fun _ _ -> Sample.normal 0.0 1.0 Distribution.rnd |> float32)
            Array.init k (fun _ -> randomGaussianMatrix ())
        {
            new IFeatureMap<Vector<float32>> with
                member __.FeatureMap x =  
                    let V = Array.map (fun w -> w * x) W
                    Array.Parallel.init nFeatures (fun i ->
                        Array.fold (fun state (v : Vector<_>) -> state * v.[i]) Z V
                    )
                    |> DenseVector.raw
                member __.Dimension = nFeatures
        }

    ///Randomized approximation to the gaussian kernel feature map.
    let randomFourierFeatureMap (sigma : float32) inputDimension numberOfFeatures = 
        let sigma = float sigma
        let normalization = sqrt (2.0f / float32 numberOfFeatures)
        let projectionMatrix = DenseMatrix.init numberOfFeatures inputDimension  (fun _ _ -> Sample.normal 0.0 (1.0 / sigma) Distribution.rnd |> float32 )
        let offSets = DenseVector.init numberOfFeatures (fun _ -> Sample.continuousUniform 0.0 (2.0 * System.Math.PI) Distribution.rnd |> float32)
        
        {
            new IFeatureMap<_> with
                member __.FeatureMap (x : Vector<float32>) = 
                    projectionMatrix * x
                    |>! Vector.mapiInPlace (fun i x -> normalization * cos(x + offSets.[i]))
                member __.Dimension = numberOfFeatures
        }

    let internal extractHerd (X : Matrix<float32>) (herd : Herd<int>) = 
            let instances = herd.Instances |> Seq.map (fun i -> X.Row(i) )
            herd.Weights, instances

    type GaussianFeatureMap (sigma : float32, instances : seq<Vector<float32>>) = 
        let instanceMatrix = DenseMatrix.ofRowSeq instances
        let norms2 = instances |> Seq.map (fun v -> v * v) |> Seq.toArray
        let g sigma x = exp( -x / (Two * pown sigma 2))

        member this.Dimension = instanceMatrix.RowCount
        member this.Instances = instanceMatrix |> Matrix.toRowSeq

        static member Collect(herds: seq<GaussianFeatureMap>) = 
            let instances = 
                herds
                |> Seq.collect (fun h -> h.Instances)

            let sigma = Seq.head herds |> fun x -> x.Sigma
            GaussianFeatureMap(sigma, instances)

        member this.Sigma = sigma

        member this.FeatureMap (x' : Vector<float32>) =
            let norm2x = x' * x'
            instanceMatrix * x'
            |>! Vector.mapiInPlace(fun i ip_rep_x ->
                g sigma (norms2.[i] + norm2x - Two * ip_rep_x) 
            ) 

        interface IFeatureMap<Vector<float32>> with
            member this.FeatureMap (x : Vector<float32>) = this.FeatureMap x
            member this.Dimension = this.Dimension

        member this.KernelMatrix() = 
            instanceMatrix.TransposeAndMultiply(instanceMatrix)
            |>! Matrix.mapiInPlace (fun i j ip -> 
                g sigma (norms2.[i] + norms2.[j] - Two * ip)  
            )
        
        member this.NormSquared (weights : Vector<float32>) = 
            let kernelMatrix = this.KernelMatrix()
            (weights * kernelMatrix * weights)

        member this.Normalize(weights : Vector<float32>) = 
            let norm = this.NormSquared(weights) |> sqrt
            weights |> Vector.mapInPlace (fun x -> x / norm)

        member this.SubSample(numberOfSamples, weights : Vector<float32>) : Vector<float32> * GaussianFeatureMap =
            let totalWeight = weights.L1Norm() |> float32
            let indices = 
                let w = weights |> Vector.asArray |> Array.map abs 
                Array.init numberOfSamples (fun _ -> Distribution.sampleCategorical w)
            let newX = indices |> Seq.map (fun i -> instanceMatrix.Row(i))
            let newWeights = 
                indices 
                |> Array.map (fun i -> if weights.[i] > Zero then totalWeight / float32 numberOfSamples else -totalWeight / float32 numberOfSamples ) 
                |> Vector.asVector
            newWeights, GaussianFeatureMap(sigma, newX) 
        
        member this.Cover(error) : GaussianFeatureMap =  
            let distanceSquared(i, j) = 
                let distance = norms2.[i] + norms2.[j] - Two * instanceMatrix.Row(i) * instanceMatrix.Row(j)
                Two - g sigma distance 
                     
            let n = this.Dimension
            let initialHerd : Herd<int> = 
                {Weights = [||] ; Instances = [|0 ..  n - 1|] }
            let error = pown error 2

            initialHerd
            |> FarthestFirstCover distanceSquared error n 
            |> Seq.map (fun i ->
                instanceMatrix.Row(i)
            )
            |> fun x -> GaussianFeatureMap(this.Sigma, x)

        member this.ClusteredSubSample(error, numberOfPartitions, maximumSize, weights : Vector<float32>) : Vector<float32> * GaussianFeatureMap =                                                         
            let distanceSquared(i, j) = 
                let distance = norms2.[i] + norms2.[j] - Two * instanceMatrix.Row(i) * instanceMatrix.Row(j)
                let weightSimilariy = let s = weights.[i] * weights.[j] in if s > Zero then One else -One
                Two - weightSimilariy * g sigma distance
       
            let initialHerd : Herd<int> = 
                {Weights = weights |> Vector.asArray ; Instances = [|0 .. weights.Count - 1|]}
            let error = pown error 2

            initialHerd
            |> ClusteredSubsamplingParallel distanceSquared error numberOfPartitions maximumSize
            |> (fun h ->
                let newWeights, newX = extractHerd instanceMatrix h
                Vector.asVector newWeights, GaussianFeatureMap(sigma, newX)
            )
    let gaussianFeatureMap sigma instances = GaussianFeatureMap(sigma, instances)
    
    module GaussianTools = 
        ///Assigns equal weights to each x_i in the feature map.
        let equalWeights (h : GaussianFeatureMap) = 
            let n = h.Dimension
            DenseVector.create n (1.0f / float32 n), h

        let collectNystromFeatureMaps (herds : seq<GaussianFeatureMap>) = GaussianFeatureMap.Collect herds
        let subSample numberOfSamples (weights, h : GaussianFeatureMap) = h.SubSample (numberOfSamples, weights)

        ///Performs clustered sub sampling as in "An Average Classification Algorithm" Section 7
        let clusteredSubSample (error, numberOfPartitions, maximumSize) (weights, h : GaussianFeatureMap) = h.ClusteredSubSample(error, numberOfPartitions, maximumSize, weights)
        let cover error (h : GaussianFeatureMap) = h.Cover(error)

        let projection (f2 : GaussianFeatureMap) (w : Vector<float32>, f1 : GaussianFeatureMap)  =
            if f1.Sigma = f2.Sigma then
                let K = f2.KernelMatrix()
                let Z = 
                    f2.Instances
                    |> Seq.map (fun x -> w * f1.FeatureMap(x) )
                    |> DenseVector.ofSeq
                K.SolveIterative(Z,MathNet.Numerics.LinearAlgebra.Single.Solvers.BiCgStab())
            else
                failwith "Sigma must be the same for both f1 and f2."
        let reduce error (w , f) =
            let c = cover error f
            let w = projection c (w ,f)
            w, c

        let NadarayaWatson sigma epsilon train = 
            let n = single (Seq.length train)
            if epsilon > Zero then 
                let baseFeatures = 
                    train 
                    |> Seq.map fst
                    |> gaussianFeatureMap sigma
            
                let reducedFeatures = 
                    baseFeatures
                    |> cover epsilon
                
                let w1 = 
                    baseFeatures
                    |> equalWeights
                    |> projection reducedFeatures
            
                let w2 = 
                    let weights = 
                        train 
                        |> Seq.map (fun x -> (snd x) / n )
                        |> DenseVector.ofSeq
                    (weights, baseFeatures)
                    |> projection reducedFeatures
            
                {
                    new IPredictor<'x> with
                        member this.Predict x = 
                            let f = reducedFeatures.FeatureMap x
                            [|(w2 * f) / (w1 * f)|] |> DenseVector.raw
                }
            else
                let baseFeatures = 
                    train 
                    |> Seq.map fst
                    |> gaussianFeatureMap sigma
            
                let w1 = 
                    baseFeatures
                    |> equalWeights
                    |> fst
            
                let w2 = 
                    train 
                    |> Seq.map (fun x -> (snd x) / n )
                    |> DenseVector.ofSeq
            
                {
                    new IPredictor<'x> with
                        member this.Predict x = 
                            let f = baseFeatures.FeatureMap x
                            [|(w2 * f) / (w1 * f)|] |> DenseVector.raw
                }

        let kernelMeanClassifier sigma epsilon train = 
            let n = single <| Seq.length train
            if epsilon > Zero then 
                let weights = 
                    train
                    |> Seq.map (fun (x, y) -> if y = 1 then One / n else -One / n)
                    |> DenseVector.ofSeq
                let baseFeatures =
                    train
                    |> Seq.map fst
                    |> gaussianFeatureMap sigma
                let w, f = 
                    (weights, baseFeatures)
                    |> reduce epsilon
                {
                    new IPredictor<Vector<float32>> with
                        member this.Predict x = 
                            [|w * f.FeatureMap(x)|] |> DenseVector.raw
                }
            else 
                let weights = 
                    train
                    |> Seq.map (fun (x, y) -> if y = 1 then One / n else -One / n)
                    |> DenseVector.ofSeq
                let baseFeatures =
                    train
                    |> Seq.map fst
                    |> gaussianFeatureMap sigma
                {
                    new IPredictor<Vector<float32>> with
                        member this.Predict x = 
                            [|weights * baseFeatures.FeatureMap(x)|] |> DenseVector.raw
                }