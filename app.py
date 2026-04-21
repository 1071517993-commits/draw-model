import pandas as pd
import numpy as np
import streamlit as st
try:
st.title("⚽ AI平局预测系统")

# =========================
# 上传数据
# =========================
uploaded_file = st.file_uploader("上传比赛数据CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("请上传CSV文件（包含赔率数据）")
    st.stop()

# =========================
# 特征工程
# =========================
df['p_home'] = 1 / df['odds_home']
df['p_draw'] = 1 / df['odds_draw']
df['p_away'] = 1 / df['odds_away']

p_sum = df['p_home'] + df['p_draw'] + df['p_away']
df['p_draw_norm'] = df['p_draw'] / p_sum

df['odds_std'] = df[['odds_home','odds_draw','odds_away']].std(axis=1)

# =========================
# 模型（简化版）
# =========================
import joblib

model = joblib.load("model.pkl")

# 特征工程（必须一致）
df['p_home'] = 1 / df['odds_home']
df['p_draw'] = 1 / df['odds_draw']
df['p_away'] = 1 / df['odds_away']

p_sum = df['p_home'] + df['p_draw'] + df['p_away']
df['p_home_norm'] = df['p_home'] / p_sum
df['p_draw_norm'] = df['p_draw'] / p_sum
df['p_away_norm'] = df['p_away'] / p_sum

df['odds_diff'] = abs(df['odds_home'] - df['odds_away'])
df['odds_std'] = df[['odds_home','odds_draw','odds_away']].std(axis=1)

X = df[['odds_home','odds_draw','odds_away',
        'p_home_norm','p_draw_norm','p_away_norm',
        'odds_diff','odds_std']]

# 🔥 三分类概率
probs = model.predict_proba(X)

df['prob_home'] = probs[:,0]
df['prob_draw'] = probs[:,1]
df['prob_away'] = probs[:,2]
# =========================
# 🔥 加入EV判断
df['EV_home'] = df['prob_home'] * df['odds_home']
df['EV_draw'] = df['prob_draw'] * df['odds_draw']
df['EV_away'] = df['prob_away'] * df['odds_away']
# =========================
# 推荐筛选
# =========================
picks = df[
    (df['prob'] > 0.38) &
    (df['odds_draw'].between(2.8, 3.3)) &
    (df['odds_diff'] < 1.2)
]

# =========================
# 展示
# =========================
st.dataframe(df[['match','prob_home','prob_draw','prob_away']])
except Exception as e:
    st.error(f"程序报错：{e}")
