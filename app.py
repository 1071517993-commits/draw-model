import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(layout="wide")
st.title("⚽")

# =========================
# 加载模型
# =========================
model = joblib.load("model_v3.pkl")

# =========================
# TAB
# =========================
tab1, tab2, tab3 = st.tabs(["📥 数据输入", "📊 预测结果", "📚 历史记录"])

# =========================
# 📥 Excel输入
# =========================
with tab1:

    st.subheader("📥 Excel式批量输入（可复制粘贴）")

    default_df = pd.DataFrame({
        "match": ["Arsenal vs Chelsea"],

        "AvgH":[2.3], "AvgD":[3.2], "AvgA":[3.1],
        "AvgCH":[2.1], "AvgCD":[3.3], "AvgCA":[3.4],

        "B365H":[2.25], "B365CH":[2.05],
        "WHH":[2.35], "WHCH":[2.15],
        "PSH":[2.28], "PSCH":[2.08],
        "BWH":[2.30], "BWCH":[2.10],

        "AHh":[-0.5], "AHCh":[-0.75],

        "B365AHH":[0.95], "B365CAHH":[1.05],
        "B365AHA":[0.95], "B365CAHA":[0.85],

        "Avg>2.5":[1.95], "AvgC>2.5":[2.10]
    })

    df_input = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True
    )

    run = st.button("🚀 一键预测")

# =========================
# 📊 预测
# =========================
with tab2:

    if run:

        df = df_input.copy()

        # ===== 数据清洗 =====
        for col in df.columns:
            if col != "match":
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=["AvgCH","AvgCD","AvgCA"])

        if len(df) == 0:
            st.error("❌ 数据为空，请检查输入")
            st.stop()

        # =========================
        # 特征工程
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

        df['B365_drop'] = df['B365H'] - df['B365CH']
        df['WH_drop'] = df['WHH'] - df['WHCH']
        df['PS_drop'] = df['PSH'] - df['PSCH']
        df['BW_drop'] = df['BWH'] - df['BWCH']

        df['market_spread'] = df[['B365CH','WHCH','PSCH','BWCH']].max(axis=1) - df[['B365CH','WHCH','PSCH','BWCH']].min(axis=1)
        df['b365_bias'] = df['B365CH'] - df['AvgCH']

        df['ah_diff'] = df['AHCh'] - df['AHh']
        df['ah_up'] = (df['ah_diff'] > 0).astype(int)
        df['ah_down'] = (df['ah_diff'] < 0).astype(int)

        df['water_home_diff'] = df['B365CAHH'] - df['B365AHH']
        df['water_away_diff'] = df['B365CAHA'] - df['B365AHA']

        df['fake_strength'] = ((df['ah_diff'] > 0) & (df['water_home_diff'] > 0)).astype(int)

        df['ou_diff'] = df['AvgC>2.5'] - df['Avg>2.5']
        df['low_scoring'] = (df['AvgC>2.5'] < 1.90).astype(int)

        df['conflict'] = ((df['ah_diff'] > 0) & (df['home_drop'] < 0)).astype(int)

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
        # 模型预测
        # =========================
        probs = model.predict_proba(X)

        df['prob_home'] = probs[:,0]
        df['prob_draw'] = probs[:,1]
        df['prob_away'] = probs[:,2]

        # =========================
        # EV
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
        # 今日最优组合
        # =========================

        # S级
        S = df[(df['signal_score'] >= 4) & (df['best_EV'] >= 1.10)]

        # A级
        A = df[(df['signal_score'] >= 3) & (df['best_EV'] >= 1.05)]

        # 排序
        S = S.sort_values(by='best_EV', ascending=False)
        A = A.sort_values(by='best_EV', ascending=False)

        # 组合策略
        combo = pd.concat([
            S.head(2),
            A.head(2)
        ]).drop_duplicates()    

        # 控制最多4场
        combo = combo.head(4)
        
        # =========================
        # 强信号
        # =========================
        df['signal_score'] = 0
        df.loc[df['best_EV'] > 1.08, 'signal_score'] += 2
        df.loc[df['market_spread'] < 0.15, 'signal_score'] += 1
        df.loc[abs(df['ah_diff']) < 0.25, 'signal_score'] += 1
        df.loc[df['low_scoring'] == 1, 'signal_score'] += 1

        # =========================
        # 凯利
        # =========================
        bankroll = st.number_input("💰 当前资金", value=1000)
        kelly_fraction = 0.25

        def kelly(p, odds):
            b = odds - 1
            q = 1 - p
            k = (b * p - q) / b
            return max(k, 0)

        df['kelly'] = df.apply(lambda x: {
            '主胜': kelly(x['prob_home'], x['AvgCH']),
            '平局': kelly(x['prob_draw'], x['AvgCD']),
            '客胜': kelly(x['prob_away'], x['AvgCA'])
        }[x['best_pick']], axis=1)

        df['bet_size'] = df['kelly'] * kelly_fraction * bankroll
        df['bet_size'] = df['bet_size'].clip(upper=bankroll * 0.1)

        # =========================
        # 展示结果
        # =========================
        show_cols = [
            'match','best_pick','best_EV','signal_score','bet_size',
            'prob_home','prob_draw','prob_away'
        ]

        st.subheader("📊 全部预测")
        st.dataframe(df[show_cols])

        st.subheader("🔥 强信号推荐")
        st.dataframe(df[df['signal_score']>=3][show_cols])
       
        st.subheader("🔥 今日最优组合（推荐下注）")

        if len(combo) > 0:
            st.dataframe(combo[['match','best_pick','best_EV','signal_score','bet_size']])
        else:
            st.write("❌ 今日无优质组合（建议不下注）")

        # =========================
        # 保存历史
        # =========================
        file = "history.csv"

        if os.path.exists(file):
            history = pd.read_csv(file)
            history = pd.concat([history, df])
        else:
            history = df

        history.to_csv(file, index=False)

# =========================
# 📚 历史
# =========================
with tab3:

    st.subheader("📚 历史记录")

    if os.path.exists("history.csv"):
        history = pd.read_csv("history.csv")

        st.dataframe(history.tail(50))

        st.download_button(
            "📥 下载历史记录",
            data=open("history.csv", "rb"),
            file_name="history.csv"
        )
    else:
        st.write("暂无记录")
