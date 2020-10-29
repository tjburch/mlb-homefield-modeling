# mlb-homefield-modeling
Using Bayesian inference to understand home-field advantage. 

Analysis code used in my blog post [Home-field Advantage in 2020](http://tylerjamesburch.com/blog/baseball/homefield-2020).

## Setting up

To setup the directory structure as designed, run:

```
make setup
```

To install appropriate packages, run the following command, ideally in a virtual environment

```
pip install -r requirements.txt
```

## Running the analysis

The analysis is set and ready to go with one command,

```
make run_base_analysis
```

This has 3 steps:

1. Download data from FiveThirtyEight, skim to the desired year range (passed by the `-y` flag, and saves to a csv in the `data/` directory)
2. Fits a model for each year, then creates all yearly plots - run expectation, home and away run distributions, and home field advantage. This is done in the script `python/standard_model.py`. Each iteration also saves an Arviz InferenceData file with the model's artifacts - these files run about 500 mb each for the 20,000 posterior samples.
3. Compares year-to-year models, in the `python/multiyear_analysis.py` file, generating a forest plot for the home-field advantage model parameter and for the actual value on the outcome variable.

If you want to do different years, or more samples just change the parameters on top of the `Makefile`. Note that the SAMPLES is per-core, this analysis was ran on a dual core laptop so simulations are 2*SAMPLES. If you run on a different number of cores, you might consider reducing the SAMPLES or increasing the number of simulations