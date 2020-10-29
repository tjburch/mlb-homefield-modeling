import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
from standard_model import simulate_seasons
from utils import load_data
from collections import namedtuple
import sys

trace_dir = sys.argv[1]
SAMPLES=int(sys.argv[2])

def load_traces():
    trace_dict = {}
    for y in range(2019,2021):
        trace_dict[y] = az.from_netcdf(f"{trace_dir}/{str(y)}.cd")
    return trace_dict

def make_forestplot(trace_dict: dict):
    """Plot beta3 parameter for each year"""
    az.plot_forest(
        data=list(trace_dict.values()),
        model_names=list(trace_dict.keys()),
        var_names=["β3"],
        hdi_prob=.95,
        combined=True,
        colors="dodgerblue",
        figsize=(6,7),
        textsize=18,
    )
    plt.savefig("plots/multiyear_beta.png")

def simulate_all_years(trace_dict: dict):
    """Simulate and plot home field advantage parameter for each year"""
    # Load all data
    data_df = load_data(dataframe=True)
    data_df = data_df.drop(["team_string"], axis=1)
    Preds = namedtuple("sims","mean_home_scores mean_away_scores league_home_scores league_away_scores")
    predictions = {}
    for year, trace in trace_dict.items():
        print(f"Doing year {year}")        
        year_data = data_df[ data_df["season"] == year-2000] 
        mean_home_scores, mean_away_scores, league_home_scores, league_away_scores = simulate_seasons(trace, year_data, year=year, simulations=SAMPLES*2)
        predictions[year] = Preds(
            mean_home_scores=mean_home_scores,
            mean_away_scores=mean_away_scores,
            league_home_scores=league_home_scores,
            league_away_scores=league_away_scores
        )    
    d ={k: v.mean_home_scores-v.mean_away_scores for k,v in predictions.items()}
    az.plot_forest(
        d,
        hdi_prob=0.95,
        colors="dodgerblue",
        figsize=(6,7),
        textsize=18,
    )
    plt.axvline(x=0, linestyle="--", color="grey")
    plt.xlabel("Home Field Run Advantage",fontsize=16)
    plt.savefig("plots/multiyear_hfa.png")


if __name__ == "__main__":
    
    # Load trace dictionary
    trace_dict = load_traces()
    # Make forestplot for beta parameter
    make_forestplot(trace_dict)
    # Simulate home-field advantage for each year and plot 
    simulate_all_years(trace_dict)

