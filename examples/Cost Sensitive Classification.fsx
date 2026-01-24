(**
Here we show how FLearn can be used to learn a cost sensitive classifier for the UCI Adult Census Data set. We show two approaches,
one using logloss with a cost sensitive threshold, and the other using a cost sensitive hinge loss.
*)

#r @"../bin/Release/net7.0/FLearn.dll"
#r "nuget: XPlot.Plotly"
#r "nuget: MathNet.Numerics.FSharp"
#r "nuget: XPlot.Plotly"

open System
open FLearn
open FLearn.PerformanceMeasures
open MathNet.Numerics.LinearAlgebra

//Data -- You will need to change this to point to a local version of the Adult Census UCI data set to replicate my results.
let path = @"C:\Users\brend\OneDrive\Data\Adult Data Set\"
let trainPath = path + "adult_train.svm"
let testPath = path + "adult_test.svm"

//Set the expected dimension of the input data and parse lib SVM format data to a vector and label.
let d = 105

let parse (s: string) =
    let blocks = s.TrimEnd(' ').Split ' '

    let label =
        match blocks.[0] with
        | "-1" -> 0
        | "1" -> 1
        | _ -> failwith "Invalid Label"

    let featureVector = DenseVector.zero d
    featureVector.[d - 1] <- 1.0f

    for s in blocks.[1..] do
        let stuff = s.Split ':'
        let index = int stuff.[0]
        let value = float32 stuff.[1]
        featureVector.[index] <- value

    featureVector, label

//Define the train and test sets, noting that data is NOT read into main memory. This is a feature of FLearn.
let train = IO.File.ReadLines trainPath |> Seq.map parse
let test = IO.File.ReadLines testPath |> Seq.map parse

//Binary Classification - Here we define the loss function, features and learner.
let cost = 5.0f //Cost for false positives.
let loss = logLoss 2 //Log loss with 2 classes.
let features = identityFeatureMap d // The "do nothing" feature map, for data that is ready to go without processing.
let learner = createAdaGradLearner 0.1f loss features // This sets the learning algorithm up, AdaGrad in this case.

learnSequence train learner // Now we learn the training sequence using our learner.

//Evaluate performance on the test set.s

let performance =
    Seq.fold
        (fun (c, l) (x, y) ->
            let p = (learner.Predict x)[1] //Probability of positive class.
            let py = if p > cost / (1.0f + cost) then 1 else 0 //Post processing thresholding.

            if y = 1 && py = 0 then c + 1.0f, l + 1.0f //False negative
            elif y = 0 && py = 1 then c + 1.0f, l + cost //False positive
            else c + 1.0f, l) //True positive or true negative

        (0.0f, 0.0f)
        test
    |> fun (c, l) -> l / c

printfn "Log loss with post processing performance: %f" performance

//Now we try a cost sensitive hinge loss.
let lossHinge = canonicalCSHinge cost //The cost sensitive hinge loss.
let learnerHinge = createAdaGradLearner 0.1f lossHinge features // This sets the learning algorithm up, AdaGrad in this case.

let trainBinary =
    train |> Seq.map (fun (x, y) -> x, if y = 1 then Positive else Negative)

let testBinary =
    test |> Seq.map (fun (x, y) -> x, if y = 1 then Positive else Negative)


learnSequence trainBinary learnerHinge
let performanceHinge = lossBatch lossHinge testBinary learnerHinge
printfn "Cost Sensitive Hinge Loss Performance: %f" performanceHinge
