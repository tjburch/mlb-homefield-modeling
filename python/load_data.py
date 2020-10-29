import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse
from utils import standardize
# Get Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-y","--year", type=int, default=2000, help="Year to start sample at")
parser.add_argument("-d","--savedir", type=str, required=True, help="Save directory")
args = parser.parse_args()
SAVEDIR=args.savedir

# Get Data from source
df = pd.read_csv("https://projects.fivethirtyeight.com/mlb-api/mlb_elo.csv")

# Slim DF to desired regular seasons seasons
df_skim = df[df["playoff"].isna()] # Not playoffs
df_skim = df_skim[df_skim["neutral"]==False] # No neural sites
df_skim = df_skim[df_skim["season"] >= args.year] # 2000 and beyond
df_skim.to_csv(SAVEDIR+"/full_elo.csv")

# Setup encoder
le = LabelEncoder()
le.fit(df_skim["team1"])

# Make home DF
first_season = df_skim["season"].min()
home_df = pd.DataFrame()
home_df["season"] = df_skim["season"] - first_season
home_df["score"] = df_skim["score1"] 
home_df["adjusted_elo"] = standardize(df_skim["elo1_pre"])
home_df["opponent_pitcher_elo"] = standardize(df_skim["pitcher2_adj"])
home_df["team_string"] = df_skim["team1"]
home_df["team"] = le.transform(df_skim["team1"])
home_df["stadium"] = le.transform(df_skim["team1"])
home_df["home"] = True

# Make away DF
away_df = pd.DataFrame()
away_df["season"] = df_skim["season"] - first_season
away_df["score"] = df_skim["score2"] 
away_df["adjusted_elo"] = standardize(df_skim["elo2_pre"])
away_df["opponent_pitcher_elo"] = standardize(df_skim["pitcher1_adj"])
away_df["team_string"] = df_skim["team2"]
away_df["team"] = le.transform(df_skim["team2"])
away_df["stadium"] = le.transform(df_skim["team1"])
away_df["home"] = False

# Join 
final_df = pd.concat([home_df, away_df])
final_df = final_df.dropna()
final_df["home"] = final_df["home"].astype(int)
final_df["score"] = final_df["score"].astype(int)
final_df["stadium"] = final_df["stadium"].astype(int)

# Save 
final_df.to_csv(SAVEDIR+"/model_df.csv")
