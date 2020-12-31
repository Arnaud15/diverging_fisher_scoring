# When Fisher Scoring diverges
- Using [Fisher scoring to fit L1-penalized logistic regression](https://www.jstatsoft.org/article/view/v033i01) models can diverge, even if the dataset is not perfectly separable.
- It's because it relies on Newton, the corresponding quadratic approximation of the loss landscape holding only locally.
- Thus, we diverge if we start from to far from the optimum.
- It's typically the case on example synthetic datasets showcased in here.
