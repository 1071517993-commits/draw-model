import pandas as pd
import numpy as np
import streamlit as st

st.title("⚽ 平局预测系统")

df = pd.DataFrame({
    "match": ["A vs B","C vs D"],
    "odds_home":[2.5,1.8],
    "odds_draw":[3.1,3.3],
    "odds_away":[2.8,4.2]
})

df['p_home'] = 1 / df['odds_home']
df['p_draw'] = 1 / df['odds_draw']
df['p_away'] = 1 / df['odds_away']

p_sum = df['p_home'] + df['p_draw'] + df['p_away']
df['p_draw_norm'] = df['p_draw'] / p_sum

df['odds_std'] = df[['odds_home','odds_draw','odds_away']].std(axis=1)

df['prob'] = df['p_draw_norm'] * 0.6 + (1 / (1 + df['odds_std'])) * 0.4
df['EV'] = df['prob'] * df['odds_draw']

st.dataframe(df)