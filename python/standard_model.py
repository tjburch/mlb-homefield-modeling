from os import makedirs
import pymc3 as pm
import numpy as np
from utils import load_data, nan_buffer
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns
from scipy.stats import mode
import argparse
import os


# Slice data
def slice_data(season):
    """Downloads data, skims to desired season"""
    data_df = load_data(dataframe=True)
    data_df = data_df.drop(["team_string"], axis=1)
    data = data_df[ data_df["season"] == season] 
    return data 

# Build and sample model
def build_model(data):
    """Builds pymc3 model, samples, returns all modeling artifacts for saving"""
    n_stadium = len(np.unique(data["stadium"]))
    standard = pm.Model()
    with standard:

        # Priors
        α = pm.Normal("α",mu=1, sigma=1)
        β1 = pm.Normal("β1", mu=0, sigma=1)
        β2 = pm.Normal("β2", mu=0, sigma=1)
        β3 = pm.Normal("β3", mu=0, sigma=0.1)
        α2 = pm.Uniform("α2", 0, 10)
        τ = pm.Normal("τ", mu=0, sigma=0.1, shape=n_stadium)

        observed_scores = pm.Data("observed_scores", data["score"])

        # Linear Regression
        μ1 = pm.math.exp(α + β1 * data["adjusted_elo"] + β2 * data["opponent_pitcher_elo"] + β3 * data["home"] + τ[data["stadium"]] )
        score = pm.NegativeBinomial("score", mu=μ1, alpha=α2, observed=observed_scores)
        prior = pm.sample_prior_predictive()
        trace = pm.sample(args.samples, chains=2, cores=2, init="advi", n_init=25000, return_inferencedata=True)
        ppc = pm.sample_posterior_predictive(trace, var_names=["score","α","α2","β1","β2","β3","τ"], samples=args.samples*2)
        # Add ppc for saving
        trace.extend(az.from_pymc3(posterior_predictive=ppc))

    return standard, prior, trace, ppc


def check_trace(trace):
    """Make trace convergence plot"""
    az.plot_trace(trace, compact=True);
    plt.savefig(f"{plot_dir}/trace_check")
    plt.close()

def simulate_runs(ppc):
    """Simulates run scoring distribution from posterior predictive check"""
    # Model check by simulating run distribution
    bincounts = []
    max_len = ppc["score"].max()+1
    sims = args.samples
    run_dist = np.zeros((sims, max_len))
    for i in range(sims):
        games = ppc["score"][i,:]
        bincounts = np.pad( np.bincount(games), pad_width=(0, max_len-len(np.bincount(games))), mode="constant", constant_values=0)
        run_dist[i,:] = bincounts
    return run_dist

def evaluate_run_distribution(run_dist):
    """Derives 1 and 2 sigma bands, MAP estimate for the run distribution"""
    max_len = len(run_dist[0])
    one_sigma = np.zeros((2,max_len))
    two_sigma = np.zeros((2,max_len))
    mle = np.zeros(max_len)
    for run_count in range(max_len):
        one_sigma[:,run_count] = np.quantile(run_dist[:,run_count], [.15865, .84135])
        two_sigma[:,run_count] = np.quantile(run_dist[:,run_count], [.02275, .97725])
        mle[run_count] = mode(run_dist[:,run_count])[0]
    return one_sigma, two_sigma, mle
    
def plot_run_distribution(one_sigma, two_sigma, mle, data, year):
    """Plots run distribution plot based on calculated 1/2 sigma bands and MAP value """
    x = np.arange(0,len(two_sigma[0,:]))
    plt.figure(figsize=(8,6))
    plt.fill_between(x, y1=two_sigma[0,:], y2=two_sigma[1,:], color="goldenrod", alpha=.7, label="$2\sigma$")
    plt.fill_between(x, y1=one_sigma[0,:], y2=one_sigma[1,:], color="forestgreen", alpha=.7, label="$1\sigma$")
    plt.plot(x, mle, color="black", alpha=0.6, marker=None, linestyle="-", label="Maximum a Posteriori")

    binned_data = np.histogram(data["score"], bins=np.arange(0,max(data["score"])))[0]
    errors = np.sqrt(binned_data)
    plt.errorbar(x=np.arange(len(binned_data)), y=binned_data, xerr=None, yerr=errors,
                linestyle="none", marker="o",
                color="black", alpha=0.7, label=f"{args.year} Value")

    plt.xlim(0,20)
    plt.ylim(bottom=0)
    plt.legend(frameon=False, fontsize=14)

    sns.despine()
    plt.xticks(np.arange(0,20))
    plt.tick_params(labelsize=12)
    plt.xlabel("Runs Scored",fontsize=14)
    plt.ylabel("Game Count", fontsize=14);
    plt.savefig(f"{plot_dir}/run_distribution")


def simulate_seasons(trace, data, year, simulations):
    """Simulates seasons given posterior parameter predictions"""
    ppc = trace.posterior_predictive

    # Expected number of games  
    expected_home = max(data.groupby("team").home.agg("sum"))
    expected_away = max(data.groupby("team").home.agg("count") - data.groupby("team").home.agg("sum"))
    expected_games = expected_home +expected_away

    # Number of teams
    n_teams = len(np.unique(data["team"]))
    
    # Empty arrays to get filled as teams are simulated
    all_home_scores = np.zeros((simulations, expected_home, n_teams))
    all_away_scores = np.zeros((simulations, expected_away, n_teams))
    
    # Iterate over teams and simulate for each
    for ht in np.arange(0,n_teams):
        
        # Skim df to current team
        team_df = data[data["team"]==ht]
        home_games = team_df[team_df["home"] == 1]
        nhome_games = len(home_games)
        away_games = team_df[team_df["home"] == 0]
        naway_games = len(away_games)
        
        #Home Game Scores - make array of dim (simulations, games)
        # todo - surely there's a nicer way to do this, but want to keep as a matrix otherwise starts to be very slow
        home_only_mu = pm.math.exp(
            ppc["α"][0:simulations].data.repeat(nhome_games).reshape((simulations,nhome_games)) + 
            ppc["β1"][0:simulations].data.repeat(nhome_games).reshape((simulations,nhome_games)) * home_games["adjusted_elo"].to_numpy().repeat(simulations).reshape((simulations,nhome_games)) +
            ppc["β2"][0:simulations].data.repeat(nhome_games).reshape((simulations,nhome_games)) * home_games["opponent_pitcher_elo"].to_numpy().repeat(simulations).reshape((simulations,nhome_games)) +
            ppc["β3"][0:simulations].data.repeat(nhome_games).reshape((simulations,nhome_games)) * 1 +  
            ppc["τ"][0,0:simulations, home_games["stadium"]].data
        )
    
        # Away Game Scores
        away_only_mu = pm.math.exp(
            ppc["α"][0:simulations].data.repeat(naway_games).reshape((simulations,naway_games)) + 
            ppc["β1"][0:simulations].data.repeat(naway_games).reshape((simulations,naway_games)) * away_games["adjusted_elo"].to_numpy().repeat(simulations).reshape((simulations,naway_games)) +
            ppc["β2"][0:simulations].data.repeat(naway_games).reshape((simulations,naway_games)) * away_games["opponent_pitcher_elo"].to_numpy().repeat(simulations).reshape((simulations,naway_games)) +
            (ppc["β3"][0:simulations].data * 0).repeat(naway_games).reshape((simulations,naway_games)) +  
            ppc["τ"][0,0:simulations, away_games["stadium"]].data
        )

        # Now that we have parameters, sample from negative binomial distribution
        # pymc3 distribution sampling doesn't give larger output shapes so use numpy, which uses a different parameter set:
        # https://en.wikipedia.org/wiki/Negative_binomial_distribution (Alternative Formulation of negative binomial)
        # Transform both home and away values to parameters for numpy sampling

        α2_home = ppc["α2"][0:simulations].data.repeat(nhome_games).reshape((simulations,nhome_games))
        p_h = (α2_home) / (α2_home + home_only_mu)
        n_h = α2_home

        α2_away = ppc["α2"][0:simulations].data.repeat(naway_games).reshape((simulations,naway_games))
        p_a = (α2_away) / (α2_away + away_only_mu)
        n_a = α2_away

        # Sample distribution with new paramters
        home_scores = np.random.negative_binomial(n=n_h, p=p_h.eval())
        away_scores = np.random.negative_binomial(n=n_a, p=p_a.eval())
        home_scores = nan_buffer(arr=home_scores, expected_shape=(simulations, expected_home), dim=1)
        away_scores = nan_buffer(arr=away_scores, expected_shape=(simulations, expected_away), dim=1)        
        all_home_scores[:,:,ht] = home_scores
        all_away_scores[:,:,ht] = away_scores

    # aggregate scores over teams to full league
    league_home_scores = all_home_scores.reshape(-1,n_teams*all_home_scores.shape[1])
    league_away_scores = all_away_scores.reshape(-1,n_teams*all_away_scores.shape[1])
    # Average
    mean_home_scores = np.nanmean(league_home_scores, axis=1)
    mean_away_scores = np.nanmean(league_away_scores, axis=1)
    return mean_home_scores, mean_away_scores, league_home_scores, league_away_scores

def plot_splits(mean_home_scores, mean_away_scores):
    """Plot a histogram with home and away simulated scores separated"""

    plt.figure(figsize=(8,6))
    plt.hist(mean_home_scores, bins=np.linspace(3,6.5,40), color="cornflowerblue", alpha=0.6, label="Average Home Score")
    plt.hist(mean_away_scores, bins=np.linspace(3,6.5,40), color="firebrick", alpha=0.6, label="Average Away Score")
    plt.legend(frameon=False,fontsize=12)
    plt.xlabel("Average Runs Scored", fontsize=14)
    plt.ylabel("Realized Seasons", fontsize=14)

    plt.ylim(bottom=0)
    plt.legend(frameon=False, fontsize=14)

    sns.despine()
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/home_away_split")

def plot_home_field_advantage(mean_home_scores, mean_away_scores, data_df):
    """Plot difference of home and away simulated mean scores as a measure of home field advantage"""
    adv = [h-a for h,a in zip(mean_home_scores,mean_away_scores)]
    plt.figure(figsize=(6,6))
    mean_val = np.array(adv).mean().round(3)
    one_sig = np.quantile(adv,[.16,.84])
    two_sig = np.quantile(adv, [0.025,0.975])
    two_sig = [round(x,3) for x in two_sig]

    histo = plt.hist(adv, bins=30, color="green", alpha=0.6)
    plt.ylim(bottom=0, top=histo[0].max() * 1.15)
    sns.despine()
    plt.tick_params(labelsize=12)

    plt.ylabel("Realized Seasons", fontsize=14)
    plt.xlabel("Home Field Run Advantage",fontsize=14)

    true_val = data_df[data_df["home"]==True]["score"].mean() - data_df[data_df["home"]==False]["score"].mean()
    plt.axvline(true_val, color="black", linestyle="--")

    delta = 0.05
    plt.annotate(f"True value = {true_val.round(3)}",(0.02,0.96), ha="left", xycoords="axes fraction", fontsize=14)
    plt.annotate(f"$\mu$ = {mean_val}",(0.02,0.96-delta), ha="left", xycoords="axes fraction", fontsize=14)
    plt.annotate(f"95% CI = [{two_sig[0]}, {two_sig[1]}]",(0.02,0.96-2*delta), ha="left", xycoords="axes fraction", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/home_field_advantage")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y","--year", type=int, help="Year")
    parser.add_argument("-s","--samples", type=int, help="Samples")
    args = parser.parse_args()

    parent_dir = os.getcwd()
    SEASON = args.year - 2000
    
    # Make saving directories
    plot_dir = f"{parent_dir}/plots/{args.year}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(f"{parent_dir}/data/"):
        os.makedirs(f"{parent_dir}/data/")
    trace_save_name = f"{parent_dir}/data/{args.year}.cd"

    # Load data and get this season
    year_data = slice_data(SEASON)
    # Build model in pymc3
    standard, prior, trace, ppc = build_model(year_data)
    # Purge existing traces if present
    if os.path.exists(trace_save_name):
        os.remove(trace_save_name)
    # Save trace to netcdf file for reloading (NOTE: these are about 500mb/20,000 samples)
    trace.to_netcdf(filename=trace_save_name)
    # Make trace convergence plot
    check_trace(trace)
    # Make run distribution plot
    run_distribution = simulate_runs(ppc)
    # Evaluate 1, 2 sigma bands and MAP value
    one_sig, two_sig, mle = evaluate_run_distribution(run_distribution)
    # Plot above values
    plot_run_distribution(one_sig, two_sig, mle, year_data, args.year)
    # Simulate new seasons using sampled parameters
    mean_home_scores, mean_away_scores, league_home_scores, league_away_scores = simulate_seasons(trace, year_data, year=args.year, simulations=args.samples*2)
    # Plot home/away split histogram
    plot_splits(mean_home_scores, mean_away_scores)
    # Plot home field advantage value
    plot_home_field_advantage(mean_home_scores, mean_away_scores, year_data)

