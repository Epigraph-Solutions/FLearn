namespace FLearn

///Data Set Simplification via Covers and Clustered Sub Sampling.
module Covers =   
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.LinearAlgebra.Single

    let internal FarthestFirstCover (metric : ('x * 'x) -> float32) (allowedError : float32) (maximumNumberofPartitions : int) (sample : 'x []) =
        let distancesToSelectedPoints = Array.create sample.Length (Infinity, 0)
        let index = [|0 .. sample.Length - 1|]
        let rec inner oldIndex error iterations = 
                let newIndex, newError = 
                    index
                    |> Array.argmax(fun i ->
                        let oldDistance = fst distancesToSelectedPoints.[i]
                        if oldDistance = Zero then Zero
                        else
                            let newDistance = metric(sample.[i], sample.[oldIndex])
                            if newDistance < oldDistance then 
                                distancesToSelectedPoints.[i] <- newDistance, oldIndex
                                newDistance
                            else
                                oldDistance
                    )
                if error / float32 iterations < allowedError || iterations >= maximumNumberofPartitions then ()
                else inner newIndex newError (iterations + 1)
        do inner 0 Infinity 1
        
        index
        |> Array.groupBy (fun i ->
            snd distancesToSelectedPoints.[i]
        )
        |> Array.map (fun (i, _ ) -> sample.[i])
    let internal FarthestFirstPartition (metric : ('x * 'x) -> float32) (allowedError : float32) (maximumNumberofPartitions : int) (sample : 'x []) =
        let maximumNumberofPartitions = if maximumNumberofPartitions = 0 then sample.Length else maximumNumberofPartitions
        let distancesToSelectedPoints = Array.create sample.Length (Infinity, 0)
        let index = [|0 .. sample.Length - 1|]
        let rec inner oldIndex error iterations = 
                let newIndex, newError = 
                    index
                    |> Array.argmax(fun i ->
                        let oldDistance = fst distancesToSelectedPoints.[i]
                        if oldDistance = Zero then Zero
                        else
                            let newDistance = metric(sample.[i], sample.[oldIndex])
                            if newDistance < oldDistance then 
                                distancesToSelectedPoints.[i] <- newDistance, oldIndex
                                newDistance
                            else
                                oldDistance
                    )
                if error / float32 iterations < allowedError || iterations >= maximumNumberofPartitions then ()
                else inner newIndex newError (iterations + 1)
        do inner 0 Infinity 1
        
        index
        |> Array.groupBy (fun i ->
            snd distancesToSelectedPoints.[i]
        )
        |> Array.map (fun (_, sub_sample) ->
            sub_sample |> Array.map (fun i -> sample.[i])
        )

    let gaussianKernelCover error sigma (sample : Vector<float32> seq) = 
        let sample = Seq.toArray sample
        let norms2 = sample |> Array.map (fun v -> v * v)
        let g sigma x = exp( -x / (Two * pown sigma 2))
        
        let distanceSquared(i, j) = 
            let distance = norms2.[i] + norms2.[j] - Two * sample.[i] * sample.[j]
            Two - g sigma distance 
        
        let memoizedDistanceSquared = memoize distanceSquared
           
        let n = Seq.length sample
        let error = pown error 2

        [|0 .. n - 1|]
        |> FarthestFirstCover memoizedDistanceSquared error n 
        |> Array.map (fun i ->
            sample.[i]
        )

    let gaussianKernelCoverParallel chunkSize error sigma sample =
        let sample = Seq.toArray sample
        sample
        |> Array.chunkBySize chunkSize
        |> Array.Parallel.map (gaussianKernelCover error sigma)
        |> Array.concat