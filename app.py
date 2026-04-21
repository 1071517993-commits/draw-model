import streamlit as st
import pandas as pd
import joblib

st.title("⚽ AI足彩预测系统（胜平负）")

try:
    uploaded_file = st.file_uploader("上传CSV文件（包含赔率）")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # ===== 检查必要列 =====
        required_cols = ['odds_home','odds_draw','odds_away']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"缺少必要列: {col}")
                st.stop()

        # ===== 特征工程 =====
        df['p_home'] = 1 / df['odds_home']
        df['p_draw'] = 1 / df['odds_draw']
        df['p_away'] = 1 / df['odds_away']

        p_sum = df['p_home'] + df['p_draw'] + df['p_away']

        df['p_home_norm'] = df['p_home'] / p_sum
        df['p_draw_norm'] = df['p_draw'] / p_sum
        df['p_away_norm'] = df['p_away'] / p_sum

        df['odds_diff'] = abs(df['odds_home'] - df['odds_away'])
        df['odds_std'] = df[['odds_home','odds_draw','odds_away']].std(axis=1)

        # ===== 加载模型 =====
        model = joblib.load("model.pkl")

        # ===== 预测 =====
        X = df[['odds_home','odds_draw','odds_away',
                'p_home_norm','p_draw_norm','p_away_norm',
                'odds_diff','odds_std']]

        probs = model.predict_proba(X)

        df['prob_home'] = probs[:,0]
        df['prob_draw'] = probs[:,1]
        df['prob_away'] = probs[:,2]

        # ===== EV计算 =====
        df['EV_home'] = df['prob_home'] * df['odds_home']
        df['EV_draw'] = df['prob_draw'] * df['odds_draw']
        df['EV_away'] = df['prob_away'] * df['odds_away']

        # ===== 自动选择最佳方向 =====
        df['best_EV'] = df[['EV_home','EV_draw','EV_away']].max(axis=1)
        df['best_pick'] = df[['EV_home','EV_draw','EV_away']].idxmax(axis=1)

        # 显示中文
        df['best_pick'] = df['best_pick'].map({
            'EV_home': '主胜',
            'EV_draw': '平局',
            'EV_away': '客胜'
        })

        # ===== 筛选推荐（核心策略） =====
        picks = df[df['best_EV'] > 1.05]

        # ===== 展示 =====
        st.subheader("📊 全部比赛预测")
        st.dataframe(df)

        st.subheader("🔥 推荐比赛（高EV）")
        st.dataframe(picks)

except Exception as e:
    st.error(f"程序报错：{e}")
