namespace FLearn

[<AutoOpenAttribute>]
module DomainTypes =
    open MathNet.Numerics.LinearAlgebra

    type Distribution<'X> =

        { Sample: unit -> 'X }

        static member inline Map(f: 'a -> 'b, p: Distribution<'a>) = { Sample = fun () -> p.Sample() |> f }
        static member inline Return(a) = { Sample = fun () -> a }

        static member inline (>>=)(p: Distribution<'a>, T: 'a -> Distribution<'b>) =
            { Sample = fun () -> let a = p.Sample() in (T a).Sample() }

        static member inline Join(p: Distribution<Distribution<'a>>) =
            { Sample = fun () -> p.Sample().Sample() }

        static member inline Zip(p1: Distribution<'a>, p2: Distribution<'b>) =
            { Sample =
                fun () ->
                    let a = p1.Sample()
                    let b = p2.Sample()
                    a, b }

        static member inline Unzip(p: Distribution<'a * 'b>) =
            { Sample = fun () -> p.Sample() |> fst }, { Sample = fun () -> p.Sample() |> snd }

        static member inline (<*>)(f: Distribution<'a -> 'b>, p: Distribution<'a>) =
            { Sample =
                fun () ->
                    let f = f.Sample()
                    let a = p.Sample()
                    f a }

        static member inline (+)(p1: Distribution<'a>, p2: Distribution<'a>) =
            { Sample = fun () -> p1.Sample() + p2.Sample() }

        static member inline (+)(a1: 'a, p2: Distribution<'a>) = { Sample = fun () -> a1 + p2.Sample() }
        static member inline (+)(p1: Distribution<'a>, a2: 'a) = { Sample = fun () -> p1.Sample() + a2 }

        static member inline (-)(p1: Distribution<'a>, p2: Distribution<'a>) =
            { Sample = fun () -> p1.Sample() - p2.Sample() }

        static member inline (-)(p1: Distribution<'a>, a2: 'a) = { Sample = fun () -> p1.Sample() - a2 }
        static member inline (-)(a1: 'a, p2: Distribution<'a>) = { Sample = fun () -> a1 - p2.Sample() }

        static member inline (*)(p1: Distribution<'a>, p2: Distribution<'a>) =
            { Sample = fun () -> p1.Sample() * p2.Sample() }

        static member inline (*)(a1: 'a, p2: Distribution<'a>) = { Sample = fun () -> a1 * p2.Sample() }
        static member inline (*)(p1: Distribution<'a>, a2: 'a) = { Sample = fun () -> p1.Sample() * a2 }

        static member inline (/)(p1: Distribution<'a>, p2: Distribution<'a>) =
            { Sample = fun () -> p1.Sample() / p2.Sample() }

        static member inline (/)(p1: Distribution<'a>, a2: 'a) = { Sample = fun () -> p1.Sample() / a2 }
        static member inline (/)(a1: 'a, p2: Distribution<'a>) = { Sample = fun () -> a1 / p2.Sample() }
        static member inline (~-)(p1: Distribution<'a>) = { Sample = fun () -> - p1.Sample() }

    type Weights = Matrix<float32>
    type Prediction = Vector<float32>
    type Gradient = Vector<float32>

    type FeatureVector = Vector<float32>

    type IFeatureMap<'instance> =
        abstract Dimension: int
        abstract FeatureMap: 'instance -> FeatureVector

    type LossFunction<'label> =
        { Dimension: int
          Loss: 'label -> Prediction -> float32 * Gradient }

    type Projection = Weights -> unit

    type IPredictor<'instance> =
        abstract Predict: 'instance -> Prediction

    type ILearner<'instance, 'label> =
        abstract LossFunction: LossFunction<'label>
        abstract Count: float32
        abstract AverageLoss: float32
        abstract PredictOnline: 'instance -> Prediction
        abstract Learn: ('instance * 'label) -> unit
        abstract TailAverage: unit -> unit
