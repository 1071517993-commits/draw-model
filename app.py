import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(layout="wide")
st.title("⚽ AI足彩交易系统 V3 Pro")

# =========================
# 加载模型
# =========================
model = joblib.load("model_v3.pkl")

# =========================
# TAB布局
# =========================
tab1, tab2, tab3 = st.tabs(["📥 比赛录入", "📊 预测结果", "📚 历史记录"])

# =========================
# 📥 输入
# =========================
with tab1:

    st.subheader("批量录入比赛")

    n_matches = st.number_input("输入比赛数量", 1, 10, 3)

    input_data = []

    for i in range(int(n_matches)):
        st.markdown(f"### 第{i+1}场")

        match = st.text_input(f"比赛名称{i}", f"Match {i}", key=f"m{i}")

        AvgH = st.number_input(f"AvgH_{i}", 1.01, 10.0, 2.3, key=f"ah{i}")
        AvgD = st.number_input(f"AvgD_{i}", 1.01, 10.0, 3.2, key=f"ad{i}")
        AvgA = st.number_input(f"AvgA_{i}", 1.01, 10.0, 3.1, key=f"aa{i}")

        AvgCH = st.number_input(f"AvgCH_{i}", 1.01, 10.0, 2.1, key=f"ach{i}")
        AvgCD = st.number_input(f"AvgCD_{i}", 1.01, 10.0, 3.3, key=f"acd{i}")
        AvgCA = st.number_input(f"AvgCA_{i}", 1.01, 10.0, 3.4, key=f"aca{i}")

        AHh = st.number_input(f"AHh_{i}", -5.0, 5.0, -0.5, key=f"ahh{i}")
        AHCh = st.number_input(f"AHCh_{i}", -5.0, 5.0, -0.75, key=f"ahc{i}")

        B365H = st.number_input(f"B365H_{i}", value=2.25, key=f"b365h{i}")
        B365CH = st.number_input(f"B365CH_{i}", value=2.05, key=f"b365ch{i}")

        WHH = st.number_input(f"WHH_{i}", value=2.35, key=f"whh{i}")
        WHCH = st.number_input(f"WHCH_{i}", value=2.15, key=f"whch{i}")

        PSH = st.number_input(f"PSH_{i}", value=2.28, key=f"psh{i}")
        PSCH = st.number_input(f"PSCH_{i}", value=2.08, key=f"psch{i}")

        BWH = st.number_input(f"BWH_{i}", value=2.30, key=f"bwh{i}")
        BWCH = st.number_input(f"BWCH_{i}", value=2.10, key=f"bwch{i}")

        B365AHH = st.number_input(f"B365AHH_{i}", value=0.95, key=f"bahh{i}")
        B365AHA = st.number_input(f"B365AHA_{i}", value=0.95, key=f"baha{i}")

        B365CAHH = st.number_input(f"B365CAHH_{i}", value=1.05, key=f"bcahh{i}")
        B365CAHA = st.number_input(f"B365CAHA_{i}", value=0.85, key=f"bcaha{i}")

        OU_open = st.number_input(f"OU_open_{i}", value=1.95, key=f"ouo{i}")
        OU_close = st.number_input(f"OU_close_{i}", value=2.10, key=f"ouc{i}")

        input_data.append({
            "match": match,
            "AvgH": AvgH, "AvgD": AvgD, "AvgA": AvgA,
            "AvgCH": AvgCH, "AvgCD": AvgCD, "AvgCA": AvgCA,
            "AHh": AHh, "AHCh": AHCh,
            "B365H": B365H, "B365CH": B365CH,
            "WHH": WHH, "WHCH": WHCH,
            "PSH": PSH, "PSCH": PSCH,
            "BWH": BWH, "BWCH": BWCH,
            "B365AHH": B365AHH, "B365AHA": B365AHA,
            "B365CAHH": B365CAHH, "B365CAHA": B365CAHA,
            "Avg>2.5": OU_open, "AvgC>2.5": OU_close
        })

    run = st.button("🚀 开始预测")

# =========================
# 📊 预测
# =========================
with tab2:

    if run:

        df = pd.DataFrame(input_data)

        # ===== 特征工程 =====
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

        # ===== 预测 =====
        probs = model.predict_proba(X)

        df['prob_home'] = probs[:,0]
        df['prob_draw'] = probs[:,1]
        df['prob_away'] = probs[:,2]

        # ===== EV =====
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

        # ===== 强信号 =====
        df['signal_score'] = 0
        df.loc[df['best_EV'] > 1.08, 'signal_score'] += 2
        df.loc[df['market_spread'] < 0.15, 'signal_score'] += 1
        df.loc[abs(df['ah_diff']) < 0.25, 'signal_score'] += 1
        df.loc[df['low_scoring'] == 1, 'signal_score'] += 1

        # ===== 凯利 =====
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

        # ===== 展示 =====
        show_cols = [
            'match','best_pick','best_EV','signal_score','bet_size',
            'prob_home','prob_draw','prob_away'
        ]

        st.subheader("📊 全部预测")
        st.dataframe(df[show_cols])

        st.subheader("🔥 强信号推荐")
        st.dataframe(df[df['signal_score']>=3][show_cols])

        # ===== 保存 =====
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

    st.subheader("历史记录")

    if os.path.exists("history.csv"):
        history = pd.read_csv("history.csv")
        st.dataframe(history.tail(50))

        st.download_button(
            "下载记录",
            data=open("history.csv","rb"),
            file_name="history.csv"
        )
    else:
        st.write("暂无记录")
