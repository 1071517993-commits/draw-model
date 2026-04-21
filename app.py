import pandas as pd
import numpy as np
import streamlit as st

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

X = df[['odds_home','odds_draw','odds_away']]
df['prob'] = model.predict_proba(X)[:,1]

df['EV'] = df['prob'] * df['odds_draw']

# =========================
# 推荐筛选
# =========================
picks = df[
    (df['prob'] > 0.35) &
    (df['odds_draw'].between(2.8, 3.3)) &
    (df['EV'] > 1.02)
]

# =========================
# 展示
# =========================
st.subheader("📊 所有比赛")
st.dataframe(df[['match','odds_draw','prob','EV']])

st.subheader("🔥 推荐平局")
st.dataframe(picks[['match','odds_draw','prob','EV']])
