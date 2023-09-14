#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: FLearn"

open System
open FLearn
open MathNet.Numerics.LinearAlgebra

//Data -- You will need to change this to point to a local version of the Adult Census UCI data set to replicate my resultslet path = @"C:\Users\User\OneDrive\Data\Adult Data Set\"
let path = @"C:\Users\User\OneDrive\Data\Adult Data Set\"
let trainPath = path + "adult.txt"
let testPath = path + "adult test.txt"

let d = 124
let parse (s : string) = 
    let blocks = s.TrimEnd(' ').Split(' ')
    let label = 
        match blocks.[0] with
        | "-1" -> 0
        | "+1" -> 1
        | _ -> failwith "Invalid Label"
    let featureVector = DenseVector.zero d

    for s in blocks.[1 ..] do
        let stuff = s.Split(':')
        let index = int stuff.[0] - 1
        let value = float32 stuff.[1]
        featureVector.[index] <- value

    featureVector.[d-1] <- 1.0f
    (featureVector, label)
let train = 
    IO.File.ReadAllLines(trainPath) 
    |> Array.map parse
let test =
    IO.File.ReadAllLines(testPath) 
    |> Array.map parse

//Logistic Regression------------------------------------------------------------------------------------------------------------
let loss = logLoss 2
let features = identityFeatureMap d
let projection = Projections.L1Ball 10.0f
let learner = createProjectedSGDLearner loss projection features 

learnSequence train learner

let performance = PerformanceMeasures.lossSummary [|loss ; zeroOneLoss 2|] test learner
learner.AverageLoss
performance.[0]
performance.[1]

learner.AverageWeights.CoerceZero(0.01)
learner.AverageWeights |> Matrix.toSeqSkipZeros |> Seq.length
learner.AverageWeights |> Matrix.iter (printfn "%A")