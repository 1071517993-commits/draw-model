import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("⚽ AI足彩预测系统 V3（盘口行为版）")

try:
    uploaded_file = st.file_uploader("上传CSV（包含盘口+赔率数据）")

    if uploaded_file is not None:

        # =========================
        # 1️⃣ 读取（自动编码）
        # =========================
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            df = pd.read_csv(uploaded_file, encoding='gbk')

        df.columns = df.columns.str.strip()

        # =========================
        # 2️⃣ 检查必要列
        # =========================
        required_cols = [
            'AvgH','AvgD','AvgA',
            'AvgCH','AvgCD','AvgCA',
            'B365H','B365CH',
            'WHH','WHCH',
            'PSH','PSCH',
            'BWH','BWCH',
            'AHh','AHCh',
            'B365AHH','B365AHA',
            'B365CAHH','B365CAHA',
            'Avg>2.5','AvgC>2.5'
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"缺少列: {missing}")
            st.stop()

        # =========================
        # 3️⃣ ===== 欧赔行为 =====
        # =========================
        df['home_drop'] = (df['AvgH'] - df['AvgCH']) / df['AvgH']
        df['draw_drop'] = (df['AvgD'] - df['AvgCD']) / df['AvgD']
        df['away_drop'] = (df['AvgA'] - df['AvgCA']) / df['AvgA']

        df['p_home'] = 1 / df['AvgCH']
        df['p_draw'] = 1 / df['AvgCD']
        df['p_away'] = 1 / df['AvgCA']

        p_sum = df['p_home'] + df['p_draw'] + df['p_away']
        df['p_home_norm'] = df['p_home'] / p_sum
        df['p_draw_norm'] = df['p_draw'] / p_sum
        df['p_away_norm'] = df['p_away'] / p_sum

        # =========================
        # 4️⃣ ===== 公司行为 =====
        # =========================
        df['B365_drop'] = df['B365H'] - df['B365CH']
        df['WH_drop'] = df['WHH'] - df['WHCH']
        df['PS_drop'] = df['PSH'] - df['PSCH']
        df['BW_drop'] = df['BWH'] - df['BWCH']

        df['market_spread'] = df[['B365CH','WHCH','PSCH','BWCH']].max(axis=1) - df[['B365CH','WHCH','PSCH','BWCH']].min(axis=1)
        df['b365_bias'] = df['B365CH'] - df['AvgCH']

        # =========================
        # 5️⃣ ===== 亚盘行为 =====
        # =========================
        df['ah_diff'] = df['AHCh'] - df['AHh']
        df['ah_up'] = (df['ah_diff'] > 0).astype(int)
        df['ah_down'] = (df['ah_diff'] < 0).astype(int)

        df['water_home_diff'] = df['B365CAHH'] - df['B365AHH']
        df['water_away_diff'] = df['B365CAHA'] - df['B365AHA']

        df['fake_strength'] = ((df['ah_diff'] > 0) & (df['water_home_diff'] > 0)).astype(int)

        # =========================
        # 6️⃣ ===== 大小球 =====
        # =========================
        df['ou_diff'] = df['AvgC>2.5'] - df['Avg>2.5']
        df['low_scoring'] = (df['AvgC>2.5'] < 1.90).astype(int)

        # =========================
        # 7️⃣ ===== 冲突信号 =====
        # =========================
        df['conflict'] = ((df['ah_diff'] > 0) & (df['home_drop'] < 0)).astype(int)

        # =========================
        # 8️⃣ 模型加载
        # =========================
        model = joblib.load("model_v3.pkl")

        features = [
            'home_drop','draw_drop','away_drop',
            'p_home_norm','p_draw_norm','p_away_norm',
            'B365_drop','WH_drop','PS_drop','BW_drop',
            'market_spread','b365_bias',
            'ah_diff','ah_up','ah_down',
            'water_home_diff','water_away_diff',
            'fake_strength',
            'ou_diff','low_scoring',
            'conflict'
        ]

        X = df[features]

        # =========================
        # 9️⃣ 预测
        # =========================
        probs = model.predict_proba(X)

        df['prob_home'] = probs[:,0]
        df['prob_draw'] = probs[:,1]
        df['prob_away'] = probs[:,2]

        # =========================
        # 🔟 EV计算
        # =========================
        df['EV_home'] = df['prob_home'] * df['AvgCH']
        df['EV_draw'] = df['prob_draw'] * df['AvgCD']
        df['EV_away'] = df['prob_away'] * df['AvgCA']

        df['best_EV'] = df[['EV_home','EV_draw','EV_away']].max(axis=1)
        df['best_pick'] = df[['EV_home','EV_draw','EV_away']].idxmax(axis=1)

        df['best_pick'] = df['best_pick'].map({
            'EV_home': '主胜',
            'EV_draw': '平局',
            'EV_away': '客胜'
        })

        # =========================
        # 11️⃣ 强信号筛选
        # =========================
        df['signal_score'] = 0

        df.loc[df['best_EV'] > 1.08, 'signal_score'] += 2
        df.loc[df['market_spread'] < 0.15, 'signal_score'] += 1
        df.loc[abs(df['ah_diff']) < 0.25, 'signal_score'] += 1
        df.loc[df['low_scoring'] == 1, 'signal_score'] += 1

        picks = df[df['signal_score'] >= 3]
        picks = picks.sort_values(by='best_EV', ascending=False)

        # =========================
        # 12️⃣ 格式优化
        # =========================
        df['prob_home'] = (df['prob_home'] * 100).round(1)
        df['prob_draw'] = (df['prob_draw'] * 100).round(1)
        df['prob_away'] = (df['prob_away'] * 100).round(1)

        df['best_EV'] = df['best_EV'].round(2)

        # =========================
        # 13️⃣ 展示
        # =========================
        show_cols = [
            'match' if 'match' in df.columns else None,
            'best_pick','best_EV','signal_score',
            'prob_home','prob_draw','prob_away'
        ]

        show_cols = [c for c in show_cols if c is not None]

        st.subheader("📊 全部比赛预测")
        st.dataframe(df[show_cols])

        st.subheader("🔥 强信号推荐（实盘用）")
        st.dataframe(picks[show_cols])

except Exception as e:
    st.error(f"程序报错：{e}")
