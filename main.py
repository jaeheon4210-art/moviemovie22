import math
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# 1. 페이지 제목 및 아이콘 설정
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")
st.title("🎬 영화 유형 나누기")

# 2. 데이터 로드[cite: 1]
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"


@st.cache_data
def load_data():
  return pd.read_csv(DATA_URL)


raw_df = load_data()

# 3. 데이터 전처리 및 파생 속성 생성[cite: 1]
# 필수 속성 결측치 제거 및 첫 주 관객이 0인 영화 제외[cite: 1]
req_cols = [
    "first_scrn",
    "first_week_audi",
    "total_audi",
    "days_in_top10",
    "movieNm",
]
df = raw_df.dropna(subset=req_cols).copy()
df = df[
    (df["first_week_audi"] > 0) & (df["first_scrn"] > 0) & (df["total_audi"] > 0)
]

# 상용로그 취하기 및 롱런 지수 계산[cite: 1]
df["first_scrn_log"] = df["first_scrn"].apply(math.log10)
df["total_audi_log"] = df["total_audi"].apply(math.log10)
df["long_run_index"] = df["total_audi"] / df["first_week_audi"]

# 4. 전체 편수와 분석 대상(묶은) 편수를 한 줄로 표시[cite: 1]
st.write(f"전체 영화: **{len(raw_df)}**편 | 분석 대상 영화: **{len(df)}**편")

# 속성 매핑 사전
feature_dict = {
    "스크린 수 (상용로그)": "first_scrn_log",
    "누적 관객 (상용로그)": "total_audi_log",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}

# 5. 군집화에 사용할 속성 선택 (기본값 4개 전체, 최소 2개)[cite: 1]
st.markdown("---")
st.subheader("⚙️ 군집화 속성 선택")
selected_feature_names = st.multiselect(
    "군집화에 사용할 속성을 선택하세요 (2개 이상 선택):",
    options=list(feature_dict.keys()),
    default=list(feature_dict.keys()),
)

if len(selected_feature_names) < 2:
  st.warning("⚠️ 군집화를 위해 최소 2개 이상의 속성을 선택해 주세요.")
  st.stop()

selected_feature_cols = [feature_dict[name] for name in selected_feature_names]

# 6. 표준화 및 K-평균 군집화 (k=3, 난수 고정)[cite: 1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[selected_feature_cols])

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster_raw"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 큰 순서대로 ㉮, ㉯, ㉰ 라벨 부여[cite: 1]
cluster_means = df.groupby("cluster_raw")["total_audi"].mean()
sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()

label_map = {
    raw_id: label for raw_id, label in zip(sorted_clusters, ["㉮", "㉯", "㉰"])
}
df["묶음"] = df["cluster_raw"].map(label_map)

# 7. 2차원 산점도[cite: 1]
st.markdown("---")
st.subheader("📊 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
  x_2d = st.selectbox(
      "2D 가로축 속성", options=list(feature_dict.keys()), index=0, key="x_2d"
  )
with col2:
  y_2d = st.selectbox(
      "2D 세로축 속성", options=list(feature_dict.keys()), index=1, key="y_2d"
  )

fig_2d = px.scatter(
    df,
    x=feature_dict[x_2d],
    y=feature_dict[y_2d],
    color="묶음",
    hover_name="movieNm",
    labels={feature_dict[x_2d]: x_2d, feature_dict[y_2d]: y_2d},
    category_orders={"묶음": ["㉮", "㉯", "㉰"]},
    title=f"2D 산점도 ({x_2d} vs {y_2d})",
)
st.plotly_chart(fig_2d, use_container_width=True)

# 8. 3차원 산점도[cite: 1]
st.markdown("---")
st.subheader("🎲 3차원 산점도")
c1, c2, c3 = st.columns(3)
with c1:
  x_3d = st.selectbox(
      "3D X축 속성", options=list(feature_dict.keys()), index=0, key="x_3d"
  )
with c2:
  y_3d = st.selectbox(
      "3D Y축 속성", options=list(feature_dict.keys()), index=1, key="y_3d"
  )
with c3:
  z_3d = st.selectbox(
      "3D Z축 속성", options=list(feature_dict.keys()), index=2, key="z_3d"
  )

fig_3d = px.scatter_3d(
    df,
    x=feature_dict[x_3d],
    y=feature_dict[y_3d],
    z=feature_dict[z_3d],
    color="묶음",
    hover_name="movieNm",
    labels={
        feature_dict[x_3d]: x_3d,
        feature_dict[y_3d]: y_3d,
        feature_dict[z_3d]: z_3d,
    },
    category_orders={"묶음": ["㉮", "㉯", "㉰"]},
    title=f"3D 산점도 ({x_3d} x {y_3d} x {z_3d})",
)
st.plotly_chart(fig_3d, use_container_width=True)

# 9. 묶음별 편수 및 네 속성의 평균(원래 단위) 표[cite: 1]
st.markdown("---")
st.subheader("📋 묶음별 요약 표 (원래 단위 평균)")

summary_df = (
    df.groupby("묶음")
    .agg(
        편수=("movieNm", "count"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객수_평균=("total_audi", "mean"),
        top10일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("long_run_index", "mean"),
    )
    .reindex(["㉮", "㉯", "㉰"])
    .reset_index()
)

summary_df.columns = [
    "묶음",
    "영화 편수",
    "스크린 수 평균",
    "누적 관객 수 평균",
    "10위권 일수 평균",
    "롱런 지수 평균",
]

# 서식 변환 (가독성 향상)
summary_df_display = summary_df.copy()
summary_df_display["스크린 수 평균"] = summary_df_display[
    "스크린 수 평균"
].apply(lambda x: f"{x:,.1f}")
summary_df_display["누적 관객 수 평균"] = summary_df_display[
    "누적 관객 수 평균"
].apply(lambda x: f"{x:,.0f}명")
summary_df_display["10위권 일수 평균"] = summary_df_display[
    "10위권 일수 평균"
].apply(lambda x: f"{x:,.1f}일")
summary_df_display["롱런 지수 평균"] = summary_df_display[
    "롱런 지수 평균"
].apply(lambda x: f"{x:,.2f}")

st.dataframe(summary_df_display, use_container_width=True)

# 10. 묶음별 누적 관객 TOP 5 영화 목록[cite: 1]
st.markdown("---")
st.subheader("🏆 묶음별 누적 관객 TOP 5 영화")

top5_cols = st.columns(3)
for idx, cluster_label in enumerate(["㉮", "㉯", "㉰"]):
  with top5_cols[idx]:
    st.markdown(f"### 묶음 {cluster_label}")
    cluster_movies = (
        df[df["묶음"] == cluster_label]
        .sort_values(by="total_audi", ascending=False)
        .head(5)
    )

    top_list = cluster_movies[["movieNm", "total_audi"]].reset_index(drop=True)
    top_list.index = top_list.index + 1
    top_list.columns = ["영화 제목", "누적 관객 수"]
    top_list["누적 관객 수"] = top_list["누적 관객 수"].apply(
        lambda x: f"{x:,.0f}명"
    )

    st.table(top_list)
