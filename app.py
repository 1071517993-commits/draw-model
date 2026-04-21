import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("⚽ AI足彩预测系统 V2（职业版）")

try:
    uploaded_file = st.file_uploader("上传CSV（需包含赔率+盘口数据）")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # =========================
        # 1️⃣ 检查必要列
        # =========================
        required_cols = [
            'AvgH','AvgD','AvgA',
            'AvgCH','AvgCD','AvgCA',
            'B365H','BWH','PSH','WHH',
            'AHh','AHCh',
            'Avg>2.5','AvgC>2.5'
        ]

        missing = [c for c in required_cols if c not in df.columns]

        if len(missing) > 0:
            st.error(f"缺少列: {missing}")
            st.stop()

        # =========================
        # 2️⃣ 特征工程（必须和训练一致）
        # =========================

        # 欧赔
        df['open_home'] = df['AvgH']
        df['open_draw'] = df['AvgD']
        df['open_away'] = df['AvgA']

        df['close_home'] = df['AvgCH']
        df['close_draw'] = df['AvgCD']
        df['close_away'] = df['AvgCA']

        df['diff_home'] = df['close_home'] - df['open_home']
        df['diff_draw'] = df['close_draw'] - df['open_draw']
        df['diff_away'] = df['close_away'] - df['open_away']

        # 概率
        df['p_home'] = 1 / df['close_home']
        df['p_draw'] = 1 / df['close_draw']
        df['p_away'] = 1 / df['close_away']

        p_sum = df['p_home'] + df['p_draw'] + df['p_away']

        df['p_home_norm'] = df['p_home'] / p_sum
        df['p_draw_norm'] = df['p_draw'] / p_sum
        df['p_away_norm'] = df['p_away'] / p_sum

        # 分歧
        df['odds_std'] = df[['B365H','BWH','PSH','WHH']].std(axis=1)

        # 亚盘
        df['ah_open'] = df['AHh']
        df['ah_close'] = df['AHCh']
        df['ah_diff'] = df['ah_close'] - df['ah_open']

        # 大小球
        df['ou_open'] = df['Avg>2.5']
        df['ou_close'] = df['AvgC>2.5']
        df['ou_diff'] = df['ou_close'] - df['ou_open']

        # =========================
        # 3️⃣ 加载模型
        # =========================
        model = joblib.load("model_v2.pkl")

        features = [
            'open_home','close_home','diff_home',
            'open_draw','close_draw','diff_draw',
            'open_away','close_away','diff_away',
            'p_home_norm','p_draw_norm','p_away_norm',
            'odds_std',
            'ah_open','ah_close','ah_diff',
            'ou_open','ou_close','ou_diff'
        ]

        X = df[features]

        # =========================
        # 4️⃣ 预测
        # =========================
        probs = model.predict_proba(X)

        df['prob_home'] = probs[:,0]
        df['prob_draw'] = probs[:,1]
        df['prob_away'] = probs[:,2]

        # =========================
        # 5️⃣ EV计算
        # =========================
        df['EV_home'] = df['prob_home'] * df['close_home']
        df['EV_draw'] = df['prob_draw'] * df['close_draw']
        df['EV_away'] = df['prob_away'] * df['close_away']

        # =========================
        # 6️⃣ 自动推荐
        # =========================
        df['best_EV'] = df[['EV_home','EV_draw','EV_away']].max(axis=1)
        df['best_pick'] = df[['EV_home','EV_draw','EV_away']].idxmax(axis=1)

        df['best_pick'] = df['best_pick'].map({
            'EV_home': '主胜',
            'EV_draw': '平局',
            'EV_away': '客胜'
        })

        # ===== 强信号评分 =====
        df['signal_score'] = 0
    
        df.loc[df['best_EV'] > 1.08, 'signal_score'] += 2
        df.loc[df['odds_std'] < 0.12, 'signal_score'] += 1
        df.loc[abs(df['ah_diff']) < 0.25, 'signal_score'] += 1
        df.loc[df['ou_diff'] < 0, 'signal_score'] += 1
    
        # ===== 筛选 =====
        picks = df[df['signal_score'] >= 3]
    
        # 排序
        picks = picks.sort_values(by='signal_score', ascending=False)
        
        # 展示
        show_cols = [
            'match',
            'best_pick',
            'best_EV',
            'signal_score',
            'prob_home','prob_draw','prob_away'
        ]

        st.subheader("🔥 强信号推荐")
        st.dataframe(picks[show_cols])
    
except Exception as e:
    st.error(f"程序报错：{e}")
