# Bayesian optimisation versus evolutionary strategies

Source: https://alexxthiery.github.io/teaching/bo-vs-es/bo-vs-es.html (page captured 2026-10-28; transcribed from the PDF for agent use, so the original page is authoritative).

The page opens with a figure of CMA-ES on a 2D function over six generations: the sampling distribution (an orange ellipse) moves, stretches and shrinks onto the optimum. The figure is not reproduced here.

## Motivation

Bayesian optimisation and evolutionary strategies use different information to guide black-box search. Bayesian optimisation constructs a probabilistic surrogate and selects evaluations through an acquisition function. CMA-ES adapts a search distribution using candidate rankings, including a covariance matrix that responds to local geometry.

Their relative performance depends on dimension, conditioning, noise, and evaluation cost. A method that requires fewer objective evaluations may spend more computation selecting them; a method that exploits local geometry may struggle with separated basins. Explaining these differences could support predictions about algorithm choice or suggest a hybrid method.

## Project goal

Investigate when Bayesian optimisation and evolutionary strategies succeed or fail on problems of your choice. You should develop an understanding of the factors that influence their performance, in what circumstances each method excels, and the trade-offs involved. Develop and test an explanation for a difference in their behaviour.

## Possible directions

- **Dimension and structure.** How do ambient dimension and effective dimension affect performance? Can either approach exploit a low-dimensional structure that is not aligned with the coordinate axes?
- **Geometry and parameterisation.** Investigate conditioning, rotations, changes of units, or boundary constraints. Which theoretical invariances survive the practical implementation?
- **Noisy evaluations.** How should the budget be divided between exploring new points and repeating evaluations? The answer may depend on the noise distribution and the rule used to select a final recommendation.
- **Multimodality.** How do population size, restarts, or acquisition choices affect the probability of finding a narrow or distant basin?
- **Evaluation cost and parallelism.** When do surrogate-fitting overhead or sequential decisions outweigh savings in function evaluations? Does the comparison change with access to parallel workers?

## Comparing methods

State your performance measure and budget, such as objective evaluations or elapsed time. Compare methods on common problems with comparable initial information and tuning effort, and explain what costs your budget includes. Repeat runs to assess variation and uncertainty in the differences between methods. For noisy objectives, assess the recommended solutions using fresh evaluations or the known noise-free objective. The lowest noisy observation can reflect a favourable error rather than a good solution.

## Reading starter kit

- A Tutorial on Bayesian Optimization. Surrogates, acquisition functions, and noisy evaluations.
- The CMA Evolution Strategy: A Tutorial. Covariance adaptation, step-size control, and invariance.
- Scalable Global Optimization via Local Bayesian Optimization. Trust-region Bayesian optimisation.
- COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting. Experimental design and performance measurement.

The page links each title; the URLs were not captured here.
