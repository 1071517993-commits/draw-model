import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request
from io import StringIO

# Selenium 可选导入（用于 OddsPortal 实时抓取）
SELENIUM_AVAILABLE = False
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    pass

st.set_page_config(layout="wide")
st.title("⚽ AI足彩交易系统 V3 Pro（Excel模式）")

# =========================
# 加载模型
# =========================
@st.cache_resource
def load_model():
    try:
        return joblib.load("model_v3.pkl")
    except:
        return None

model = load_model()

# =========================
# Football-Data.co.uk 数据获取函数
# =========================
FOOTBALL_DATA_LEAGUES = {
    "英格兰超级联赛": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "英格兰冠军联赛": "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
    "西班牙甲级联赛": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "德国甲级联赛": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "意大利甲级联赛": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "法国甲级联赛": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
    "荷兰甲级联赛": "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
    "葡萄牙超级联赛": "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
}

def fetch_football_data(league_name):
    """从 football-data.co.uk 获取历史赔率数据"""
    if league_name not in FOOTBALL_DATA_LEAGUES:
        return None
    
    url = FOOTBALL_DATA_LEAGUES[league_name]
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode('utf-8')
        
        df = pd.read_csv(StringIO(data))
        
        required_cols = ['HomeTeam', 'AwayTeam']
        if not all(col in df.columns for col in required_cols):
            return None
        
        df = df.dropna(subset=['HomeTeam', 'AwayTeam'])
        
        matches_data = []
        
        for _, row in df.iterrows():
            match_data = {
                "match": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                "time": str(row.get('Date', '')),
                "league": league_name,
            }
            
            # 赔率数据
            if 'AvgH' in df.columns and pd.notna(row.get('AvgH')):
                match_data['AvgH'] = row['AvgH']
                match_data['AvgD'] = row['AvgD']
                match_data['AvgA'] = row['AvgA']
            elif 'B365H' in df.columns and pd.notna(row.get('B365H')):
                match_data['AvgH'] = row.get('B365H', 0)
                match_data['AvgD'] = row.get('B365D', 0)
                match_data['AvgA'] = row.get('B365A', 0)
            else:
                continue
            
            # 闭盘赔率 (使用相同值)
            match_data['AvgCH'] = match_data.get('AvgH', 0)
            match_data['AvgCD'] = match_data.get('AvgD', 0)
            match_data['AvgCA'] = match_data.get('AvgA', 0)
            
            # 各博彩公司赔率
            match_data['B365H'] = row.get('B365H', match_data.get('AvgH', 0))
            match_data['B365D'] = row.get('B365D', match_data.get('AvgD', 0))
            match_data['B365A'] = row.get('B365A', match_data.get('AvgA', 0))
            match_data['B365CH'] = match_data['B365H']
            
            match_data['WHH'] = row.get('WHH', match_data.get('AvgH', 0))
            match_data['WHD'] = row.get('WHD', match_data.get('AvgD', 0))
            match_data['WHA'] = row.get('WHA', match_data.get('AvgA', 0))
            match_data['WHCH'] = match_data['WHH']
            
            match_data['PSH'] = row.get('PSH', match_data.get('AvgH', 0))
            match_data['PSD'] = row.get('PSD', match_data.get('AvgD', 0))
            match_data['PSA'] = row.get('PSA', match_data.get('AvgA', 0))
            match_data['PSCH'] = match_data['PSH']
            
            match_data['BWH'] = row.get('BWH', match_data.get('AvgH', 0))
            match_data['BWD'] = row.get('BWD', match_data.get('AvgD', 0))
            match_data['BWA'] = row.get('BWA', match_data.get('AvgA', 0))
            match_data['BWCH'] = match_data['BWH']
            
            # 亚盘和大小球 (默认值)
            match_data['AHh'] = 0
            match_data['AHCh'] = 0
            match_data['B365AHH'] = 0.95
            match_data['B365CAHH'] = 0.95
            match_data['B365AHA'] = 0.95
            match_data['B365CAHA'] = 0.95
            match_data['Avg>2.5'] = 1.95
            match_data['AvgC>2.5'] = 1.95
            
            # 比赛结果
            if 'FTR' in df.columns:
                match_data['result'] = row.get('FTR', '')
            if 'FTHG' in df.columns and 'FTAG' in df.columns:
                match_data['home_goals'] = row.get('FTHG', 0)
                match_data['away_goals'] = row.get('FTAG', 0)
            
            matches_data.append(match_data)
        
        return matches_data
        
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        return None

# =========================
# Oddsportal 抓取函数
# =========================
def scrape_oddsportal(league_url=None):
    """从 oddsportal.com 抓取赔率数据"""
    if not SELENIUM_AVAILABLE:
        return None, "Selenium 未安装，无法使用 OddsPortal 抓取功能"
    
    if league_url is None:
        league_url = "https://www.oddsportal.com/matches/soccer/"
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(league_url)
        
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "table-main")))
        
        matches_data = []
        rows = driver.find_elements(By.CSS_SELECTOR, "tr.deactivate")
        
        for row in rows:
            try:
                match_elem = row.find_element(By.CSS_SELECTOR, "td.name a")
                match_name = match_elem.text.strip()
                
                if not match_name or " - " not in match_name:
                    continue
                
                time_elem = row.find_element(By.CSS_SELECTOR, "td.time")
                match_time = time_elem.text.strip()
                
                odds_cells = row.find_elements(By.CSS_SELECTOR, "td.odds-nowrp")
                
                if len(odds_cells) >= 3:
                    home_odds = odds_cells[0].text.strip()
                    draw_odds = odds_cells[1].text.strip()
                    away_odds = odds_cells[2].text.strip()
                    
                    try:
                        home_odds = float(home_odds) if home_odds else 0
                        draw_odds = float(draw_odds) if draw_odds else 0
                        away_odds = float(away_odds) if away_odds else 0
                    except ValueError:
                        continue
                    
                    if home_odds > 0 and draw_odds > 0 and away_odds > 0:
                        matches_data.append({
                            "match": match_name,
                            "time": match_time,
                            "AvgH": home_odds,
                            "AvgD": draw_odds,
                            "AvgA": away_odds,
                            "AvgCH": home_odds,
                            "AvgCD": draw_odds,
                            "AvgCA": away_odds,
                            "B365H": home_odds, "B365CH": home_odds,
                            "WHH": home_odds, "WHCH": home_odds,
                            "PSH": home_odds, "PSCH": home_odds,
                            "BWH": home_odds, "BWCH": home_odds,
                            "AHh": 0, "AHCh": 0,
                            "B365AHH": 0.95, "B365CAHH": 0.95,
                            "B365AHA": 0.95, "B365CAHA": 0.95,
                            "Avg>2.5": 1.95, "AvgC>2.5": 1.95
                        })
            except:
                continue
        
        return matches_data
        
    except Exception as e:
        st.error(f"抓取失败: {str(e)}")
        return None
    finally:
        if driver:
            driver.quit()

# =========================
# TAB
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📥 数据输入", "🌐 自动抓取", "📊 预测结果", "📚 历史记录"])

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

    df_input = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)
    run = st.button("🚀 一键预测")

# =========================
# 🌐 自动抓取
# =========================
with tab2:
    st.subheader("🌐 自动抓取赔率")
    
    # 数据源选择
    data_source = st.radio("选择数据源", ["Football-Data.co.uk (历史数据)", "OddsPortal (实时赔率)"])
    
    if "Football-Data" in data_source:
        st.markdown("""
        ### Football-Data.co.uk 使用说明
        - 提供各大联赛的历史赔率数据
        - 数据包含比赛结果，可用于模型训练
        - 无需浏览器，直接下载CSV文件
        """)
        
        league_select = st.selectbox("选择联赛", list(FOOTBALL_DATA_LEAGUES.keys()))
        
        if st.button("📥 获取联赛数据", type="primary"):
            with st.spinner("正在获取数据..."):
                data = fetch_football_data(league_select)
                
                if data and len(data) > 0:
                    st.session_state['scraped_df'] = pd.DataFrame(data)
                    st.success(f"✅ 成功获取 {len(data)} 场比赛！")
                else:
                    st.warning("⚠️ 未获取到数据")
    
    else:  # OddsPortal
        if not SELENIUM_AVAILABLE:
            st.warning("⚠️ Selenium 未安装，OddsPortal 抓取功能不可用")
            st.info("如需使用此功能，请安装: `pip install selenium webdriver-manager`")
        else:
            st.markdown("""
            ### OddsPortal 使用说明
            - 抓取实时赔率数据
            - 需要安装 Chrome 浏览器
            - 可能需要 VPN
            """)
            
            custom_url = st.text_input("自定义URL（可选）", placeholder="https://www.oddsportal.com/soccer/england/premier-league/")
            
            if st.button("🔄 开始抓取赔率", type="primary"):
                with st.spinner("正在抓取..."):
                    url = custom_url if custom_url else None
                    result = scrape_oddsportal(url)
                    
                    # 处理返回值（可能是元组或列表）
                    if isinstance(result, tuple):
                        data, error_msg = result
                        if error_msg:
                            st.error(f"❌ {error_msg}")
                        elif data and len(data) > 0:
                            st.session_state['scraped_df'] = pd.DataFrame(data)
                            st.success(f"✅ 成功抓取 {len(data)} 场比赛！")
                        else:
                            st.warning("⚠️ 未抓取到数据")
                    elif result and len(result) > 0:
                        st.session_state['scraped_df'] = pd.DataFrame(result)
                        st.success(f"✅ 成功抓取 {len(result)} 场比赛！")
                    else:
                        st.warning("⚠️ 未抓取到数据")
    
    # 显示和编辑数据
    if 'scraped_df' in st.session_state:
        st.subheader("📊 数据预览")
        preview_cols = ['match', 'time', 'AvgH', 'AvgD', 'AvgA']
        available_cols = [c for c in preview_cols if c in st.session_state['scraped_df'].columns]
        st.dataframe(st.session_state['scraped_df'][available_cols], use_container_width=True)
        
        st.subheader("✏️ 编辑数据")
        edited = st.data_editor(st.session_state['scraped_df'], num_rows="dynamic", use_container_width=True, key="editor")
        st.session_state['scraped_df'] = edited
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 填充到输入表格"):
                st.session_state['df_input'] = edited.copy()
                st.success("✅ 已填充到输入表格！")
        
        with col2:
            if st.button("🚀 直接预测"):
                st.session_state['df_input'] = edited.copy()
                st.session_state['run_prediction'] = True
                st.success("✅ 请切换到「预测结果」标签")

# =========================
# 📊 预测
# =========================
with tab3:
    run_prediction = st.session_state.get('run_prediction', False)
    
    if run or run_prediction:
        if run_prediction and 'df_input' in st.session_state:
            df = st.session_state['df_input'].copy()
            st.session_state['run_prediction'] = False
        else:
            df = df_input.copy()

        # 数据清洗
        for col in df.columns:
            if col not in ["match", "time", "league", "result"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=["AvgCH","AvgCD","AvgCA"])

        if len(df) == 0:
            st.error("❌ 数据为空")
            st.stop()

        # 特征工程
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

        # 模型预测
        if model:
            probs = model.predict_proba(X)
            df['prob_home'] = probs[:,0]
            df['prob_draw'] = probs[:,1]
            df['prob_away'] = probs[:,2]
        else:
            st.warning("⚠️ 未找到模型，使用简化预测")
            df['prob_home'] = 1 / df['AvgCH'] / (1/df['AvgCH'] + 1/df['AvgCD'] + 1/df['AvgCA'])
            df['prob_draw'] = 1 / df['AvgCD'] / (1/df['AvgCH'] + 1/df['AvgCD'] + 1/df['AvgCA'])
            df['prob_away'] = 1 / df['AvgCA'] / (1/df['AvgCH'] + 1/df['AvgCD'] + 1/df['AvgCA'])

        # EV
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
        
        # 强信号
        df['signal_score'] = 0
        df.loc[df['best_EV'] > 1.08, 'signal_score'] += 2
        df.loc[df['market_spread'] < 0.15, 'signal_score'] += 1
        df.loc[abs(df['ah_diff']) < 0.25, 'signal_score'] += 1
        df.loc[df['low_scoring'] == 1, 'signal_score'] += 1

        # 凯利
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

        # 最优组合
        S = df[(df['signal_score'] >= 4) & (df['best_EV'] >= 1.10)]
        A = df[(df['signal_score'] >= 3) & (df['best_EV'] >= 1.05)]

        S = S.sort_values(by='best_EV', ascending=False)
        A = A.sort_values(by='best_EV', ascending=False)

        combo = pd.concat([S.head(2), A.head(2)]).drop_duplicates()
        combo = combo.head(4)

        # 展示结果
        show_cols = ['match','best_pick','best_EV','signal_score','bet_size','prob_home','prob_draw','prob_away']

        st.subheader("📊 全部预测")
        st.dataframe(df[show_cols])

        st.subheader("🔥 强信号推荐")
        strong_signals = df[df['signal_score']>=3]
        if len(strong_signals) > 0:
            st.dataframe(strong_signals[show_cols])
        else:
            st.info("暂无强信号推荐")
       
        st.subheader("🔥 今日最优组合")
        if len(combo) > 0:
            st.dataframe(combo[['match','best_pick','best_EV','signal_score','bet_size']])
        else:
            st.write("❌ 今日无优质组合")

        # 保存历史
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
with tab4:
    st.subheader("📚 历史记录")

    if os.path.exists("history.csv"):
        history = pd.read_csv("history.csv")
        st.dataframe(history.tail(50))
        st.download_button("📥 下载历史记录", data=open("history.csv", "rb"), file_name="history.csv")
    else:
        st.write("暂无记录")
