## FLearn, Definition

> 1. To learn through failure. I flearnt how to ski by sliding face first down a mountain.
> 2. A machine learning library written in F# for F#. Woah, FLearn learned that pattern quickly!

Built on top of the [Math.Net Numerics](https://numerics.mathdotnet.com/) linear algebra library, FLearn offers an extensible, scalable and interactive take on machine learning. Utilizing tools from online convex optimization, FLearn provides a framework for specifying and optimizing various predictive models. Straight out of the box you can:

1. Construct predictive models, with statistical and worst case guarantees, via the minimization of convex loss functions.
2. Utilize feature maps such as those provided by random fourier features and the nystrom method.
3. Make predictions and estimate their error. Think error bars and gaussian processes.
4. Learn from noisily labeled data via corruption corrected loss functions.

Example usage of the package can be found [here](https://mlunhinged.visualstudio.com/_git/FLearn?path=/examples).