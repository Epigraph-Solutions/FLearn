namespace FLearn

module Projections = 
    open FLearn.Utilities
    open MathNet.Numerics.LinearAlgebra

    let internal l2Projection radius (v : Vector<float32>) = 
        let norm = v.L2Norm() |> float32
        if norm > radius then v |> Vector.mapSkipZeros (fun x -> radius * x / norm)
        else v
    let L2Ball radius : Projection = Matrix.mapRowsInPlace (fun _ -> l2Projection radius)
   
    let internal mahalanobisProjection (K : Matrix<float32>) radius (v : Vector<float32>) = 
        let norm = v * K * v |> sqrt |> float32
        if norm > radius then v |> Vector.mapSkipZeros (fun x -> radius * x / norm)
        else v
    let MahalanobisBall K radius : Projection = Matrix.mapRowsInPlace (fun _ -> mahalanobisProjection K radius)
    
    let internal softThreshold lambda (v : Vector<float32>) = 
        v 
        |> Vector.mapSkipZeros (fun x -> 
            if abs x <= lambda then Zero
            else if x > lambda then x - lambda
            else x + lambda
        )
    let internal l1Projection radius (v : Vector<float32>) = 
        let norm = v.L1Norm() |> float32
        if norm <= radius then v
        else
            let sorted = v |> Vector.asArray |> Array.sortByDescending abs 
            let n = sorted.Length
            let rec inner i runningSum =
                if i = n - 1 then (norm - radius) / float32 n
                else
                    let lambda = (runningSum - radius) / float32 (i + 1)
                    if (lambda >= abs sorted.[i + 1] && lambda <= abs sorted.[i]) 
                        then lambda
                    else inner (i + 1) (runningSum + abs sorted.[i + 1])
            let lambda = inner 0 (abs sorted.[0])
            softThreshold lambda v
    let L1Ball radius : Projection = Matrix.mapRowsInPlace (fun _ -> l1Projection radius)