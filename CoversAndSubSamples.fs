namespace FLearn

///Data Set Simplification via Covers and Clustered Sub Sampling.
module internal CoversAndSubSamples =   
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.LinearAlgebra.Single

    type WeightedSample<'x> =  
        {
            Weights : float32 []
            Instances : 'x []
        }

        member this.Length = this.Weights.Length
    let collectWeightedSamples (weightedSamples: seq<WeightedSample<'Y>>) = 
        let weights = weightedSamples |> Seq.collect (fun x -> x.Weights) |> Seq.toArray
        let instances = weightedSamples |> Seq.collect (fun x -> x.Instances) |> Seq.toArray
        {Weights = weights ; Instances = instances}
    let partition (numberofPartitions : int) (weightedSample : WeightedSample<'x>) = 
        let weights, instances = 
            weightedSample
            |-> ((fun x -> Array.splitInto numberofPartitions x.Weights), (fun x -> Array.splitInto numberofPartitions x.Instances))
        Array.map2(fun w x -> {Weights = w ; Instances = x}) weights instances
    let reweight (weightedSample : WeightedSample<'x>) : float32 * WeightedSample<'x> =
        let totalWeight = Array.sumBy abs weightedSample.Weights
        let newWeights = weightedSample.Weights |> Array.map (fun w -> w / totalWeight)
        totalWeight, {weightedSample with Weights = newWeights}
    let subsample n (weightedSample : WeightedSample<'x>) : WeightedSample<'x> = 
        let weights = weightedSample.Weights |> Array.map abs
        let samples = Array.init n (fun _ -> Distribution.sampleCategorical weights)
        {
            Weights = 
                samples 
                |> Array.map (fun i -> if weightedSample.Weights.[i] > Zero then One / float32 n else -One / float32 n) 
            Instances = 
                samples 
                |> Array.map (fun i -> weightedSample.Instances.[i])
        }
    
    let FarthestFirstCover (metric : ('x * 'x) -> float32) (allowedError : float32) (maximumNumberofPartitions : int) (weightedSample : WeightedSample<'x>) =
        let maximumNumberofPartitions = if maximumNumberofPartitions = 0 then weightedSample.Length else maximumNumberofPartitions
        let points = weightedSample.Instances
        let distancesToSelectedPoints = Array.create points.Length (Infinity, 0)
        let index = [|0 .. points.Length - 1|]
        let rec inner oldIndex error iterations = 
                let newIndex, newError = 
                    index
                    |> Array.argmax(fun i ->
                        let oldDistance = fst distancesToSelectedPoints.[i]
                        if oldDistance = Zero then Zero
                        else
                            let newDistance = metric(points.[i], points.[oldIndex])
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
        |> Array.map (fun (i, _ ) -> points.[i])
    let FarthestFirstPartition (metric : ('x * 'x) -> float32) (allowedError : float32) (maximumNumberofPartitions : int) (weightedSample : WeightedSample<'x>) =
        let maximumNumberofPartitions = if maximumNumberofPartitions = 0 then weightedSample.Length else maximumNumberofPartitions
        let weights, points = weightedSample.Weights, weightedSample.Instances
        let distancesToSelectedPoints = Array.create points.Length (Infinity, 0)
        let index = [|0 .. points.Length - 1|]
        let rec inner oldIndex error iterations = 
                let newIndex, newError = 
                    index
                    |> Array.argmax(fun i ->
                        let oldDistance = fst distancesToSelectedPoints.[i]
                        if oldDistance = Zero then Zero
                        else
                            let newDistance = metric(points.[i], points.[oldIndex])
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
        |> Array.map (fun (_, sub_weightedSample) ->
            let totalWeight = sub_weightedSample |> Array.sumBy (fun i -> abs weights.[i])
            let weights = sub_weightedSample |> Array.map (fun i -> weights.[i] / totalWeight)
            let instances = sub_weightedSample |> Array.map (fun i -> points.[i])
            totalWeight, {Weights = weights ; Instances = instances} 
        )
       
    let ClusteredSubsampling totalWeight (metric : ('x * 'x) -> float32) allowedError sparsityGoal (weightedSample : WeightedSample<'x>) : WeightedSample<'x> =             
        let partition = FarthestFirstPartition metric allowedError (sparsityGoal / 2) weightedSample
        let nCover = partition.Length
        [|
            for weight, h in partition do
                let n = (int << ceil) (weight * float32 nCover)
                let subWeightedSample = subsample n h
                yield 
                    {
                        Weights = 
                            subWeightedSample.Weights 
                            |> Array.map ( (*) (totalWeight * weight) )
                        Instances = subWeightedSample.Instances
                    }
        |]     
        |> collectWeightedSamples               
    let ClusteredSubsamplingParallel metric allowedError (numberofPartitions : int) sparsityGoal (weightedSample : WeightedSample<'x>) : WeightedSample<'x> = 
        if numberofPartitions < 1 then failwith "Invalid Number of Partitions"
        if numberofPartitions > 1 then
            let chunks = partition numberofPartitions weightedSample
            let sparsityGoal = sparsityGoal / numberofPartitions
            chunks
            |> Array.parallelMap (fun x -> 
                let totalWeight, x = reweight x
                ClusteredSubsampling totalWeight metric allowedError sparsityGoal x
            )
            |> collectWeightedSamples
        else
            let totalWeight, x = reweight weightedSample
            ClusteredSubsampling totalWeight metric allowedError sparsityGoal x