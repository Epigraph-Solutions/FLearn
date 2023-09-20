## FLearn

> 1. To learn through failure.
> 2. A supervised learning library written in F# for F#.

Built on top of the [Math.Net Numerics](https://numerics.mathdotnet.com/) linear algebra library, FLearn offers an extensible, scalable, and interactive take on supervised learning. Utilizing tools from online convex optimization, FLearn provides a framework for specifying and optimizing various predictive models. Straight out of the box you can:

1. Construct predictive models, with statistical and worst-case guarantees, via the minimization of convex loss functions.
2. Utilize feature maps such as those provided by random Fourier features and the Nystrom method.
3. Make predictions and estimate their errors. Think error bars and gaussian processes.
4. Learn from noisily labeled data via corruption corrected loss functions.

FLearn is both an "online" and a "batch" learning library, that is it can learn from data sequentially as it comes, or learn from a large training set of data in one go. Example usage of the package can be found [here](https://mlunhinged.visualstudio.com/_git/FLearn?path=/examples).