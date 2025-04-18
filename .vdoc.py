# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
import requests

url = "https://kenpom.com/cbbga25.txt"
file_path = "./baskteball.txt"

try:
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"File downloaded successfully to {file_path}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

#
#
#

import pandas as pd 
import numpy as np
import bambi as bmb
import arviz as az

import arviz as az
import numpy as np
import xarray as xr

#
#
#

# should write a column spec for this 
df = pd.read_fwf('baskteball.txt')
df.columns = ['date', 'team1', 'score1', 'team2', "score2", "single_char", "loc1", "loc2"]
df['full_loc'] = [x + y for x,y in zip(df['loc1'],df["loc2"])]
df.drop(columns=['loc1', 'loc2'], axis=1, inplace=True)

df['point_diff1'] = df['score1'] - df['score2']
df['point_diff2'] = df['score2'] - df['score1']

df['total_score'] = df['score1'] + df['score2']
df['winning_team'] = [team1 if score1 > score2 else team2 
                    for team1, score1, team2, score2
                    in zip(
                        df['team1'],
                        df['score1'],
                        df['team2'],
                        df['score2']
                    ) ]

teams =  df['team1'].unique().tolist() +  df['team2'].unique().tolist()
teams = list(set(teams))

res = {}
for team in teams:
    tdf = df.loc[ (df['team1'] == team) | (df['team2'] == team)]
    tdf.sort_values('date', inplace=True)
    tdf['point_diff'] = [
        point_diff2 if team == team2 else point_diff1
        for  point_diff1, point_diff2, team2 
        in zip(tdf['point_diff1'], tdf['point_diff2'], tdf['team2'])
        ] 
    tdf['win'] = [1 if team == winner else 0 for winner in tdf['winning_team']]
    tdf['cum_wins'] = np.cumsum(tdf['win'])
    tdf['games_played'] = np.arange(len(tdf)) + 1
    tdf['win_record']  = tdf["cum_wins"] / tdf['games_played']
    tdf['team'] = team
    tdf['opposing_team'] = [
        team1 if team == team2 else team2 
        for team1, team2 
        in zip(tdf['team1'], tdf['team2'])
        ]

    res[team] = tdf[ ['team', 'date', 'cum_wins', "games_played", "win_record", 'opposing_team', 'point_diff'] ]

# then join it to it itself by opposing team I think?

final_df = pd.concat(res)
opposing_df = final_df[ ['team', 'win_record', 'date','games_played'] ]
opposing_df.columns = ['opposing_team', "owr", 'date', 'ogp']
final_df = pd.merge(final_df, opposing_df, how = "left", on = ['opposing_team', 'date'])

final_df.to_csv("long.csv")

#
#
#
#
#

df = pd.read_csv('long.csv')

model = bmb.Model(
    "point_diff ~  ogp*owr + ( 1|  team)",
    df,
    categorical="team")

xr.set_options(display_expand_data=False, display_expand_attrs=False);


idata = model.fit(chains=4, cores=4, tune= 500, draws=1000)

#temp = az.summary
# can't plot a trace until after I figure out how to filter params
#az.plot_trace(idata)

idata.posterior.dims

m_0 = bmb.Model("point_diff ~  ogp*owr + ( 1|  team)", df, categorical="team")

#
#
#
