namespace FLearn

[<AutoOpenAttribute>]
module DistributionWorkflow =
    type DistributionBuilder() = 
        member inline this.Return x : Distribution<_> = Distribution<_>.Return x
        member inline this.ReturnFrom(p : Distribution<'a>) = p
        member inline this.Bind(p : Distribution<'a>, T: 'a -> Distribution<'b>) = p >>= T
        member this.Delay(f: unit -> Distribution<'a>) = f ()
    let distribution = DistributionBuilder()

///Operations for creating new distributions via the probability monad.
module Distribution =
    open FLearn.Utilities
    open MathNet.Numerics.Distributions

    let rnd = new MathNet.Numerics.Random.SystemRandomSource(threadSafe = true)

    let create f = {Sample = f}
    let always x = {Sample = (fun () -> x)}
    let sample (p: Distribution<'a>) = p.Sample()
    let samples (p: Distribution<'a>) = p |> Seq.unfold (fun p -> Some(sample p, p)) |> Seq.cache
    let map (f : 'a -> 'b) (p : Distribution<'a>) = {Sample = fun () -> let a = p.Sample() in f a}
    let map2 f p1 p2 = create (fun () -> let a, b = sample p1, sample p2 in f a b)
    let lift f x : Distribution<_> = always (f x)
    let replicate n (p: Distribution<'a>) = create (fun () -> Array.init n (fun _ -> sample p))

    let coinFlip (p : float32) = 
        let p = float p
        create (fun () -> let z = Sample.bernoulli p rnd in if z = 1 then true else false)
    let randomInteger lower upper = create (fun () -> rnd.Next(lower, upper))
    let inline sampleCategorical (weights : 't []) =
        let weights = Array.map float weights
        Sample.categorical weights rnd
    let inline mixture (weights: 't []) (ps: Distribution<'a> []) = 
        create (fun () ->
            let index = sampleCategorical weights
            sample ps.[index]
        )
    let empirical (atoms: seq<'a>) =
        let atoms = Seq.toArray atoms
        create (fun () ->
            let index = rnd.Next(atoms.Length)
            atoms.[index]
        )
    let inline empiricalWithWeight (weights: 't []) (atoms: 'a []) =
        create (fun () ->
            let index = sampleCategorical weights
            atoms.[index]
        )
    let inline estimateExpectation n (f : 'x -> 'y) distribution = 
        samples distribution
        |> Seq.take n
        |> Seq.averageBy f
    let estimateProbability n predicate distribution = 
        estimateExpectation n (fun x -> if predicate x then One else Zero) distribution