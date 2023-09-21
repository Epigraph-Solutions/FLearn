namespace FLearn

module internal Utilities =
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.LinearAlgebra.Single

    //Named Numbers
    let Zero = float32 0.0
    let OneHalf = float32 0.5
    let One = float32 1.0
    let Two = float32 2.0
    let Infinity = float32 infinity
        
    ///Imperative Pipe
    let (|>!) x f = f x |> ignore; x

    //Split Map
    let (|->) x (f, g) = f x, g x
    
    let memoize (f: 'T -> 'U) =
        let dict = new System.Collections.Concurrent.ConcurrentDictionary<_,_>(HashIdentity.Structural)
        fun x -> 
            if dict.ContainsKey(x) then dict.[x] 
            else
                let result = f x 
                dict.TryAdd(x, result) |> ignore
                result

    module Array = 
        let inline foldi f (initialState : 'state) (a : 'a []) = 
            let state = ref initialState
            let count = ref 0
            for x in a do 
                state.Value <- f count.Value state.Value x
                count.Value <- count.Value + 1
            state.Value
        let inline argmax f (a : 'a []) = 
            foldi(fun i (j, max) x ->
                let newMax = f x 
                if newMax > max then
                    (i, newMax)
                else (j, max)
            ) (0, -Infinity) a

        let parallelMap f (a : 'a []) = 
            a
            |> Seq.map (fun a -> async{return f a})
            |> Async.Parallel
            |> Async.RunSynchronously

    module Vector = 
        let asVector (x : float32 []) = x |> DenseVector.raw
        let asArray (x : Vector<float32>) = 
            try
                let v = x :?> DenseVector in v.Values
            with
            | _ -> Vector.toArray x

        let sparseOuterProduct(u : Vector<float32>, v : Vector<float32>) = 
            let rows = u.Count
            let cols = v.Count
            let u = u |> Vector.toSeqiSkipZeros
            let v = v |> Vector.toSeqiSkipZeros 
            seq{
                for (i, u) in u do
                    for (j, v) in v do
                        yield i, j, u * v
            }
            |> SparseMatrix.ofSeqi rows cols

    module Matrix = 
        open MathNet.Numerics.Providers.LinearAlgebra

        let rnd = System.Random()

        let inline denseMatrixAsValues (M : Matrix<float32>) = let M = M :?> DenseMatrix in M.Values
        let inline normSquared (M : Matrix<float32>) = 
            let sum = ref Zero
            for i in 0 .. M.RowCount - 1 do
                for j in 0 .. M.ColumnCount - 1 do
                    let x = M[i,j]
                    sum.Value <- sum.Value + x * x
            sum.Value
        let inline gradientUpdate (W : Weights, gradient : Matrix<float32>, learningRate : float32) =
            let valuesW = denseMatrixAsValues W
            let valuesGradient = denseMatrixAsValues gradient
            LinearAlgebraControl.Provider.AddVectorToScaledVector(valuesW, -One / sqrt learningRate, valuesGradient, valuesW)
        let inline adaGradUpdate (W : Weights, gradient : Matrix<float32>, learningRate : Weights, gSquared) =
            for i in 0 .. W.RowCount - 1 do
                for j in 0 .. W.ColumnCount - 1 do
                    W[i,j] <- W[i,j] - gradient.[i,j] / sqrt(gSquared + learningRate.[i,j])
        let inline runningAverage (count : float32, v : Matrix<float32>, u : Matrix<float32>) =            
            for i in 0 .. v.RowCount - 1 do
                for j in 0 .. v.ColumnCount - 1 do
                    v.[i,j] <- (count / (count + One)) * v.[i,j] + (One / (count + One)) * u.[i, j]

        let inline sparseNormSquared (M : Matrix<float32>) = 
            let sum = ref Zero 
            for x in Matrix.toSeqSkipZeros M do
                sum.Value <- sum.Value + x * x
            sum.Value
        let inline sparseDotProduct (i, v1 : Matrix<float32>, v2) = 
            let sum = ref 0.0f

            for (j, v2) in Vector.toSeqiSkipZeros v2 do
                sum.Value <- sum.Value + v2 * v1.[i,j]

            sum.Value
        let inline sparseMultiply (M : Matrix<float32>, v2) = 
            DenseVector.init M.RowCount (fun i -> sparseDotProduct(i, M, v2) )
        let inline sparseGradientUpdate (W : Weights, gradient : Weights, learningRate : float32) = 
            let Z = One / sqrt learningRate
            for (i, j, g) in Matrix.toSeqiSkipZeros gradient do
                W.[i,j] <- W.[i,j] - Z * g