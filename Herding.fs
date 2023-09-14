namespace FLearn

///Data Set Simplification via Herding.
module internal Herding =   
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.LinearAlgebra.Single

    type Herd<'x> =  
        {
            Weights : float32 []
            Instances : 'x []
        }

        member this.Length = this.Weights.Length
    let collectHerds (herds: seq<Herd<'Y>>) = 
        let weights = herds |> Seq.collect (fun x -> x.Weights) |> Seq.toArray
        let instances = herds |> Seq.collect (fun x -> x.Instances) |> Seq.toArray
        {Weights = weights ; Instances = instances}
    let partition (numberofPartitions : int) (herd : Herd<'x>) = 
        let weights, instances = 
            herd
            |-> ((fun x -> Array.splitInto numberofPartitions x.Weights), (fun x -> Array.splitInto numberofPartitions x.Instances))
        Array.map2(fun w x -> {Weights = w ; Instances = x}) weights instances
    let reweight (herd : Herd<'x>) : float32 * Herd<'x> =
        let totalWeight = Array.sumBy abs herd.Weights
        let newWeights = herd.Weights |> Array.map (fun w -> w / totalWeight)
        totalWeight, {herd with Weights = newWeights}
    let subsample n (herd : Herd<'x>) : Herd<'x> = 
        let weights = herd.Weights |> Array.map abs
        let samples = Array.init n (fun _ -> Distribution.sampleCategorical weights)
        {
            Weights = 
                samples 
                |> Array.map (fun i -> if herd.Weights.[i] > Zero then One / float32 n else -One / float32 n) 
            Instances = 
                samples 
                |> Array.map (fun i -> herd.Instances.[i])
        }
    
    let FarthestFirstCover (metric : ('x * 'x) -> float32) (allowedError : float32) (maximumNumberofPartitions : int) (herd : Herd<'x>) =
        let maximumNumberofPartitions = if maximumNumberofPartitions = 0 then herd.Length else maximumNumberofPartitions
        let points = herd.Instances
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
    let FarthestFirstPartition (metric : ('x * 'x) -> float32) (allowedError : float32) (maximumNumberofPartitions : int) (herd : Herd<'x>) =
        let maximumNumberofPartitions = if maximumNumberofPartitions = 0 then herd.Length else maximumNumberofPartitions
        let weights, points = herd.Weights, herd.Instances
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
        |> Array.map (fun (_, sub_herd) ->
            let totalWeight = sub_herd |> Array.sumBy (fun i -> abs weights.[i])
            let weights = sub_herd |> Array.map (fun i -> weights.[i] / totalWeight)
            let instances = sub_herd |> Array.map (fun i -> points.[i])
            totalWeight, {Weights = weights ; Instances = instances} 
        )
       
    let ClusteredSubsampling totalWeight (metric : ('x * 'x) -> float32) allowedError sparsityGoal (herd : Herd<'x>) : Herd<'x> =             
        let partition = FarthestFirstPartition metric allowedError (sparsityGoal / 2) herd
        let nCover = partition.Length
        [|
            for weight, h in partition do
                let n = (int << ceil) (weight * float32 nCover)
                let subHerd = subsample n h
                yield 
                    {
                        Weights = 
                            subHerd.Weights 
                            |> Array.map ( (*) (totalWeight * weight) )
                        Instances = subHerd.Instances
                    }
        |]     
        |> collectHerds               
    let ClusteredSubsamplingParallel metric allowedError (numberofPartitions : int) sparsityGoal (herd : Herd<'x>) : Herd<'x> = 
        if numberofPartitions < 1 then failwith "Invalid Number of Partitions"
        if numberofPartitions > 1 then
            let chunks = partition numberofPartitions herd
            let sparsityGoal = sparsityGoal / numberofPartitions
            chunks
            |> Array.parallelMap (fun x -> 
                let totalWeight, x = reweight x
                ClusteredSubsampling totalWeight metric allowedError sparsityGoal x
            )
            |> collectHerds
        else
            let totalWeight, x = reweight herd
            ClusteredSubsampling totalWeight metric allowedError sparsityGoal x