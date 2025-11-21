import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression  # 回归分析核心模型
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # 决策树核心模型
from sklearn.metrics import (
    accuracy_score, auc, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# 1. 페이지 기본 설정
# ----------------------
st.set_page_config(
    page_title="하이브리드모형 동적 프레임워크（의사결정나무+회귀분석）",
    page_icon="📊",
    layout="wide"
)

# 전역 상태 관리（각 단계 데이터/모델 저장，새로고침 시 손실 방지）
if "step" not in st.session_state:
    st.session_state.step = 0  # 0:초기화면 1:데이터업로드 2:데이터시각화 3:데이터전처리 4:모델학습 5:예측 6:평가
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}  # 단일 파일만 저장
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {"imputer": None, "scaler": None, "encoders": None, "feature_cols": None, "target_col": None}
if "models" not in st.session_state:
    # 模型：regression（회귀분석）、decision_tree（의사결정나무）
    st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.3, "decision_tree": 0.7}}
if "task" not in st.session_state:
    st.session_state.task = "logit"  # 기본값 logit（분류），의사결정나무（회귀）로 전환 가능

# ----------------------
# 2. 사이드바：단계导航 + 핵심 설정
# ----------------------
st.sidebar.title("📌 하이브리드모형 작업 흐름")
st.sidebar.divider()

# 단계导航 버튼（新增「데이터 시각화」단계）
steps = ["초기 설정", "데이터 업로드", "데이터 시각화", "데이터 전처리", "모델 학습", "모델 예측", "성능 평가"]
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i

# 핵심 설정（작업 유형 + 혼합 가중치）
st.sidebar.divider()
st.sidebar.subheader("핵심 설정")
st.session_state.task = st.sidebar.radio("작업 유형", options=["logit", "의사결정나무"], index=0)

if st.session_state.step >= 4:  # 모델 학습 후 가중치 조정 가능
    st.sidebar.subheader("하이브리드모형 가중치")
    reg_weight = st.sidebar.slider(
        "회귀 분석 가중치（해석력 강함）",
        min_value=0.0, max_value=1.0, value=st.session_state.models["mixed_weights"]["regression"], step=0.1
    )
    st.session_state.models["mixed_weights"]["regression"] = reg_weight
    st.session_state.models["mixed_weights"]["decision_tree"] = 1 - reg_weight
    st.sidebar.text(f"의사결정나무 가중치（정확도 높음）：{1 - reg_weight:.1f}")

# ----------------------
# 3. 메인 페이지：단계별 내용 표시
# ----------------------
st.title("📊 하이브리드모형 동적 배포 프레임워크")
st.markdown("**단일 원본 데이터 파일 업로드 후，시각화→전처리→학습→예측 전과정을 한 번에 완성**")
st.markdown("### 🧩 핵심 모델：회귀 분석（Regression）+ 의사결정나무（Decision Tree）")
st.divider()

# ----------------------
# 단계 0：초기 설정（안내 페이지）
# ----------------------
if st.session_state.step == 0:
    st.subheader("🎉 하이브리드모형 동적 프레임워크에 오신 것을 환영합니다")
    st.markdown("""
    본 프레임워크는 **데이터 수령 후 직접 업로드하여 사용**할 수 있으며，사전 전처리나 모델 학습이 필요 없습니다. 핵심 흐름은 다음과 같습니다：
    
    1. **데이터 업로드**：단일 원본 파일（CSV/Parquet/Excel）을 업로드
    2. **데이터 시각화**：범주형 변수와 수치형 변수를 선택하여 다양한 그래프로 데이터 탐색
    3. **데이터 전처리**：결측값 채우기、범주형 특징 인코딩
    4. **모델 학습**：「회귀 분석+의사결정나무」하이브리드모형 학습
    5. **모델 예측**：단일 데이터 입력 또는 일괄 업로드 예측을 지원
    6. **성능 평가**：하이브리드모형과 단일 모형의 성능을 비교
    
    ### 적용 가능场景
    - logit 작업（분류）：사용자가 서비스를 수락할지 여부、위반 여부等 이진 예측（모델：로지스틱 회귀+분류 의사결정나무）
    - 의사결정나무 작업（회귀）：판매량、금액、평점等 연속값 예측（모델：선형 회귀+회귀 의사결정나무）
    
    ### 왼쪽「데이터 업로드」를 클릭하여 사용을 시작하세요！
    """)

# ----------------------
# 단계 1：데이터 업로드（단일 파일만 업로드）
# ----------------------
elif st.session_state.step == 1:
    st.subheader("📤 데이터 업로드（단일 파일）")
    st.markdown("지원 형식：CSV、Parquet、Excel（.xlsx/.xls）")
    st.markdown("⚠️  파일에 타겟 열（예측할 변수）과 특징 열（예측에 사용할 변수）이 모두 포함되어야 합니다")
    
    # 단일 파일 업로드 컴포넌트
    uploaded_file = st.file_uploader("데이터 파일 선택", type=["csv", "parquet", "xlsx", "xls"], key="single_file")
    
    if uploaded_file is not None:
        try:
            # 다양한 형식 파일 읽기
            if uploaded_file.name.endswith(".csv"):
                df_merged = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                df_merged = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                df_merged = pd.read_excel(uploaded_file)
            else:
                st.error("지원하지 않는 파일 형식입니다！CSV/Parquet/Excel 파일을 업로드하세요")
                st.stop()
            
            # 데이터 저장
            st.session_state.data["merged"] = df_merged
            
            # 데이터 정보 표시
            st.success(f"데이터 업로드 성공！")
            st.metric("데이터 양", f"{len(df_merged):,} 행 × {len(df_merged.columns)} 열")
            st.markdown("### 데이터 미리보기")
            st.dataframe(df_merged.head(5), use_container_width=True)
            
            # 데이터 기본 정보 추가 표시
            st.markdown("### 데이터 기본 정보")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**열 이름**")
                st.write(", ".join(df_merged.columns.tolist()[:10]) + ("..." if len(df_merged.columns) > 10 else ""))
            with col2:
                st.write("**결측값 총 개수**")
                st.write(f"{df_merged.isnull().sum().sum()} 개")
            with col3:
                st.write("**데이터 유형**")
                st.write(df_merged.dtypes.value_counts().to_string())
            
            # 下一步 안내
            st.divider()
            st.info("📊 데이터 탐색을 위해 왼쪽 사이드바에서「데이터 시각화」단계로 이동하세요")
        
        except Exception as e:
            st.error(f"데이터 읽기 실패：{str(e)}")

# ----------------------
# 단계 2：데이터 시각화（新增！범주형x + 수치형y + 그래프 선택）
# ----------------------
elif st.session_state.step == 2:
    st.subheader("📊 데이터 시각화")
    
    if st.session_state.data["merged"] is None:
        st.warning("먼저「데이터 업로드」단계를 완료하세요")
    else:
        df = st.session_state.data["merged"]
        
        # 1. 변수 유형 자동识别
        st.markdown("### 변수 선택")
        # 범주형 변수（object, category）
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        # 수치형 변수（int64, float64）
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        #  변수 선택 박스（선택 가능하도록）
        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox("📋 X축：범주형 변수", options=cat_cols, index=0 if cat_cols else None, disabled=not cat_cols)
        with col2:
            y_var = st.selectbox("📈 Y축：수치형 변수", options=num_cols, index=0 if num_cols else None, disabled=not num_cols)
        with col3:
            # 그래프 타입 선택（범주형x + 수치형y에 적합한 그래프）
            graph_types = [
                "막대 그래프（평균값）", 
                "박스 플롯（분포）", 
                "바이올린 플롯（분포+밀도）",
                "산점도（개별 데이터）",
                "선 그래프（추세）"
            ]
            graph_type = st.selectbox("📊 그래프 유형", options=graph_types, index=0)
        
        # 2. 그래프 그리기（조건에 맞춰 동적 생성）
        st.divider()
        if x_var and y_var:
            st.markdown(f"### {x_var} vs {y_var} ({graph_type.split('（')[0]})")
            
            # 그래프 데이터 준비（결측값 제거）
            plot_df = df[[x_var, y_var]].dropna()
            
            try:
                # 그래프 타입에 따라 plotly 그리기
                if graph_type == "막대 그래프（평균값）":
                    # 각 범주별 y_var 평균값 계산
                    bar_data = plot_df.groupby(x_var)[y_var].mean().reset_index()
                    fig = px.bar(
                        bar_data, x=x_var, y=y_var, 
                        title=f"{x_var}별 {y_var} 평균값",
                        labels={y_var: f"{y_var} 평균값", x_var: x_var},
                        color=x_var, color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                
                elif graph_type == "박스 플롯（분포）":
                    fig = px.box(
                        plot_df, x=x_var, y=y_var,
                        title=f"{x_var}별 {y_var} 분포",
                        labels={y_var: y_var, x_var: x_var},
                        color=x_var, color_discrete_sequence=px.colors.qualitative.Set2
                    )
                
                elif graph_type == "바이올린 플롯（분포+밀도）":
                    fig = px.violin(
                        plot_df, x=x_var, y=y_var,
                        title=f"{x_var}별 {y_var} 분포 및 밀도",
                        labels={y_var: y_var, x_var: x_var},
                        color=x_var, box=True,  # 박스 플롯 포함
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                
                elif graph_type == "산점도（개별 데이터）":
                    fig = px.scatter(
                        plot_df, x=x_var, y=y_var,
                        title=f"{x_var} vs {y_var} 개별 데이터 분포",
                        labels={y_var: y_var, x_var: x_var},
                        color=x_var, opacity=0.6,
                        color_discrete_sequence=px.colors.qualitative.Vivid
                    )
                
                elif graph_type == "선 그래프（추세）":
                    # 범주형 변수를 순서대로 정렬
                    line_data = plot_df.groupby(x_var)[y_var].mean().reset_index()
                    fig = px.line(
                        line_data, x=x_var, y=y_var,
                        title=f"{x_var}별 {y_var} 추세",
                        labels={y_var: y_var, x_var: x_var},
                        color_discrete_sequence=["#1f77b4"],
                        markers=True
                    )
                
                # 그래프 스타일 최적화
                fig.update_layout(
                    width=1200, height=600,
                    xaxis_title_font=dict(size=14),
                    yaxis_title_font=dict(size=14),
                    title_font=dict(size=16, weight="bold")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 3. 통계 정보 표시
                st.markdown("### 📋 통계 정보")
                stats_df = plot_df.groupby(x_var)[y_var].agg([
                    "count", "mean", "std", "min", "25%", "50%", "75%", "max"
                ]).round(3)
                stats_df.columns = ["데이터 개수", "평균값", "표준편차", "최소값", "제1사분위수", "중앙값", "제3사분위수", "최대값"]
                st.dataframe(stats_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"그래프 생성 실패：{str(e)}")
        else:
            st.warning("범주형 변수(X)와 수치형 변수(Y)를 모두 선택해야 합니다")
        
        # 下一步 안내
        st.divider()
        st.info("🔧 데이터 전처리를 위해 왼쪽 사이드바에서「데이터 전처리」단계로 이동하세요")

# ----------------------
# 단계 3：데이터 전처리（修复 selectbox 错误）
# ----------------------
elif st.session_state.step == 3:
    st.subheader("🧹 데이터 전처리")
    
    if st.session_state.data["merged"] is None:
        st.warning("먼저「데이터 업로드」단계를 완료하세요")
    else:
        df_merged = st.session_state.data["merged"]
        
        # 1. 데이터 개요（결측값、데이터 유형）
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 데이터 기본 정보")
            st.write(f"총 데이터 양：{len(df_merged):,} 행 × {len(df_merged.columns)} 열")
            st.write("데이터 유형 분포：")
            st.dataframe(df_merged.dtypes.value_counts().reset_index(), use_container_width=True)
        
        with col2:
            st.markdown("### 결측값 분포")
            missing_info = df_merged.isnull().sum()[df_merged.isnull().sum() > 0].reset_index()
            missing_info.columns = ["필드명", "결측값 개수"]
            if len(missing_info) > 0:
                st.dataframe(missing_info, use_container_width=True)
                fig_missing = px.imshow(df_merged.isnull(), color_continuous_scale="Reds", title="결측값 히트맵")
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("결측값이 없습니다！")
        
        # 2. 전처리 설정（修复 selectbox 错误）
        st.divider()
        st.markdown("### 전처리 매개변수 설정")
        
        # 타겟 열 선택（예측 변수）- 核心修复：index=0（默认第一个列），增加有效性校验
        if len(df_merged.columns) > 0:
            target_col = st.selectbox(
                "타겟 열 선택（예측할 변수）", 
                options=df_merged.columns, 
                index=0  # 修复：默认选择第一个列，而非 -1
            )
            st.session_state.preprocess["target_col"] = target_col
        else:
            st.error("데이터에 열이 존재하지 않습니다！올바른 데이터 파일을 업로드하세요")
            st.stop()
        
        # 특징 열 선택（타겟 열과 무관한 열 제외）
        exclude_cols = st.multiselect(
            "제외할 열 선택（예：ID、무관한 필드）", 
            options=[col for col in df_merged.columns if col != target_col]
        )
        feature_cols = [col for col in df_merged.columns if col not in exclude_cols + [target_col]]
        
        # 特征列有效性校验
        if not feature_cols:
            st.warning("특징 열이 선택되지 않았습니다！제외할 열을 조정하세요")
        st.session_state.preprocess["feature_cols"] = feature_cols
        
        # 결측값 처리
        st.markdown("#### 결측값 처리")
        impute_strategy = st.selectbox("수치형 결측값 채우기 방식", options=["중앙값", "평균값", "최빈값"], index=0)
        impute_strategy_map = {"중앙값": "median", "평균값": "mean", "최빈값": "most_frequent"}
        
        # 범주형 특징 인코딩
        st.markdown("#### 범주형 특징 인코딩")
        cat_encoding = st.selectbox("범주형 특징 인코딩 방식", options=["레이블 인코딩（LabelEncoder）", "원-핫 인코딩（OneHotEncoder）"], index=0)
        
        # 3. 전처리 실행（增加特征列非空校验）
        if st.button("전처리 시작"):
            if not feature_cols:
                st.error("전처리 실패：특징 열이 없습니다！")
                continue
            
            try:
                X = df_merged[feature_cols].copy()
                y = df_merged[target_col].copy()
                
                # 수치형과 범주형 특징 분리
                num_cols = X.select_dtypes(include=["int64", "float64"]).columns
                cat_cols = X.select_dtypes(include=["object", "category"]).columns
                
                # 수치형 전처리：결측값 채우기 + 표준화
                imputer = SimpleImputer(strategy=impute_strategy_map[impute_strategy])
                X[num_cols] = imputer.fit_transform(X[num_cols])
                
                scaler = StandardScaler()
                X[num_cols] = scaler.fit_transform(X[num_cols])
                
                # 범주형 전처리：결측값 채우기 + 인코딩
                encoders = {}
                for col in cat_cols:
                    # 범주형 결측값을 "알 수 없음"으로 채우기
                    X[col] = X[col].fillna("알 수 없음").astype(str)
                    
                    if cat_encoding == "레이블 인코딩（LabelEncoder）":
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                        encoders[col] = le
                    else:  # 원-핫 인코딩
                        ohe = OneHotEncoder(sparse_output=False, drop="first")
                        ohe_result = ohe.fit_transform(X[[col]])
                        ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]  # 첫 번째 범주 제외（다중공선성 방지）
                        X = pd.concat([X.drop(col, axis=1), pd.DataFrame(ohe_result, columns=ohe_cols)], axis=1)
                        encoders[col] = (ohe, ohe_cols)
                
                # 전처리组件 저장
                st.session_state.preprocess["imputer"] = imputer
                st.session_state.preprocess["scaler"] = scaler
                st.session_state.preprocess["encoders"] = encoders
                st.session_state.preprocess["feature_cols"] = list(X.columns)  # 업데이트된 특징 열（원-핫 인코딩 열 포함）
                
                # 전처리된 데이터 저장
                st.session_state.data["X_processed"] = X
                st.session_state.data["y_processed"] = y
                
                st.success("데이터 전처리 완료！")
                st.markdown(f"전처리 후 특징 수：{len(X.columns)}")
                st.dataframe(X.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"전처리 실패：{str(e)}")

# ----------------------
# 단계 4：모델 학습（하이브리드모형：회귀 분석+의사결정나무）
# ----------------------
elif st.session_state.step == 4:
    st.subheader("🚀 하이브리드모형 학습（회귀 분석 + 의사결정나무）")
    
    # 전처리 완료 여부 확인
    if "X_processed" not in st.session_state.data or "y_processed" not in st.session_state.data:
        st.warning("먼저「데이터 전처리」단계를 완료하세요")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        # 데이터 분할（학습集+테스트集）
        st.markdown("### 학습 설정")
        test_size = st.slider("테스트集 비율", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=y if st.session_state.task == "logit" else None
        )
        
        # 모델 선택（작업 유형에 따라）
        if st.session_state.task == "logit":  # 분류任务：로지스틱 회귀（회귀분석）+ 분류 의사결정나무
            reg_model = LogisticRegression(max_iter=1000)  # 分类用 회귀분석（로지스틱）
            dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)  # 分类 의사결정나무
        else:  # 회귀任务：선형 회귀（회귀분석）+ 회귀 의사결정나무
            reg_model = LinearRegression()  # 回归用 회귀분석（선형）
            dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)  # 回归 의사결정나무
        
        # 모델 학습
        if st.button("모델 학습 시작"):
            with st.spinner("모델 학습 중..."):
                # 단일 모델 학습
                reg_model.fit(X_train, y_train)
                dt_model.fit(X_train, y_train)
                
                # 모델 저장
                st.session_state.models["regression"] = reg_model
                st.session_state.models["decision_tree"] = dt_model
                
                # 학습集/테스트集 저장
                st.session_state.data["X_train"] = X_train
                st.session_state.data["X_test"] = X_test
                st.session_state.data["y_train"] = y_train
                st.session_state.data["y_test"] = y_test
                
                st.success("모델 학습 완료！")
                st.markdown("✅ 학습된 모델：")
                st.markdown("- 회귀 분석（로지스틱/선형，해석력 강함）")
                st.markdown("- 의사결정나무（분류/회귀，정확도 높음）")
                st.markdown("- 하이브리드모형（전两者 가중融合）")

# ----------------------
# 단계 5：모델 예측（단일/일괄 업로드）
# ----------------------
elif st.session_state.step == 5:
    st.subheader("🎯 모델 예측")
    
    # 모델 학습 완료 여부 확인
    if st.session_state.models["regression"] is None or st.session_state.models["decision_tree"] is None:
        st.warning("먼저「모델 학습」단계를 완료하세요")
    else:
        # 예측 함수（전처리 로직 재사용 + 新模型适配）
        def predict(input_data):
            X = input_data.copy()
            preprocess = st.session_state.preprocess
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            
            # 수치형 전처리
            X[num_cols] = preprocess["imputer"].transform(X[num_cols])
            X[num_cols] = preprocess["scaler"].transform(X[num_cols])
            
            # 범주형 전처리
            for col in cat_cols:
                X[col] = X[col].fillna("알 수 없음").astype(str)
                encoder = preprocess["encoders"][col]
                
                if isinstance(encoder, LabelEncoder):
                    # 미본적 범주 처리
                    X[col] = X[col].replace([x for x in X[col].unique() if x not in encoder.classes_], "알 수 없음")
                    if "알 수 없음" not in encoder.classes_:
                        encoder.classes_ = np.append(encoder.classes_, "알 수 없음")
                    X[col] = encoder.transform(X[col])
                else:  # OneHotEncoder
                    ohe, ohe_cols = encoder
                    ohe_result = ohe.transform(X[[col]])
                    X = pd.concat([X.drop(col, axis=1), pd.DataFrame(ohe_result, columns=ohe_cols)], axis=1)
            
            # 특징 열 순서 일치 보장
            X = X[preprocess["feature_cols"]]
            
            # 하이브리드모형 예측（가중融合）
            reg_weight = st.session_state.models["mixed_weights"]["regression"]
            dt_weight = st.session_state.models["mixed_weights"]["decision_tree"]
            reg_model = st.session_state.models["regression"]
            dt_model = st.session_state.models["decision_tree"]
            
            if st.session_state.task == "logit":  # 분류 예측
                reg_proba = reg_model.predict_proba(X)[:, 1]  # 로지스틱 회귀 확률
                dt_proba = dt_model.predict_proba(X)[:, 1]    # 의사결정나무 확률
                mixed_proba = reg_weight * reg_proba + dt_weight * dt_proba
                pred = (mixed_proba >= 0.5).astype(int)
                return pred, mixed_proba
            else:  # 회귀 예측
                reg_pred = reg_model.predict(X)  # 선형 회귀 예측값
                dt_pred = dt_model.predict(X)    # 의사결정나무 예측값
                mixed_pred = reg_weight * reg_pred + dt_weight * dt_pred
                return mixed_pred, None
        
        # 예측 방식 선택
        predict_mode = st.radio("예측 방식", options=["단일 데이터 입력", "일괄 업로드 CSV"])
        
        # 단일 입력 예측
        if predict_mode == "단일 데이터 입력":
            st.markdown("#### 단일 데이터 입력（특징값을 입력하세요）")
            feature_cols = st.session_state.preprocess["feature_cols"]
            input_data = {}
            
            # 특징 유형에 따라 동적으로 입력 폼 생성
            with st.form("single_pred_form"):
                cols = st.columns(3)
                for i, col in enumerate(feature_cols[:9]):  # 최대 9개 특징 표시（화면 혼잡 방지）
                    with cols[i % 3]:
                        # 특징 유형 판단（수치/범주）
                        if col in st.session_state.data["X_processed"].select_dtypes(include=["int64", "float64"]).columns:
                            input_data[col] = st.number_input(col, value=0.0)
                        else:
                            # 범주형 특징：학습集中의 고유값을 옵션으로 제시
                            unique_vals = st.session_state.data["X_processed"][col].unique()[:10]  # 최대 10개 옵션
                            input_data[col] = st.selectbox(col, options=unique_vals)
                
                # 예측 제출
                submit_btn = st.form_submit_button("예측 시작")
            
            if submit_btn:
                input_df = pd.DataFrame([input_data])
                pred, proba = predict(input_df)
                
                st.divider()
                st.markdown("### 예측 결과")
                if st.session_state.task == "logit":
                    st.metric("예측 결과", "양성" if pred[0] == 1 else "음성")
                    st.metric("양성 확률", f"{proba[0]:.3f}" if proba is not None else "-")
                else:  # 의사결정나무（회귀）
                    st.metric("예측 결과", f"{pred[0]:.2f}")
        
        # 일괄 업로드 예측
        else:
            st.markdown("#### 일괄 업로드 CSV 예측")
            uploaded_file = st.file_uploader("특징 열을 포함한 CSV 파일 업로드", type=["csv"])
            
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                st.metric("업로드 데이터 양", f"{len(batch_df):,} 행")
                st.dataframe(batch_df.head(3), use_container_width=True)
                
                # 특징 열 일치 확인
                required_features = st.session_state.preprocess["feature_cols"]
                missing_features = [col for col in required_features if col not in batch_df.columns]
                if missing_features:
                    st.warning(f"업로드된 파일에 필요한 특징 열이 없습니다：{', '.join(missing_features)}")
                else:
                    if st.button("일괄 예측 시작"):
                        with st.spinner("예측 중..."):
                            pred, proba = predict(batch_df)
                            batch_df["하이브리드모형 예측 결과"] = pred
                            if proba is not None:
                                batch_df["양성 확률"] = proba.round(3)
                            
                            st.divider()
                            st.markdown("### 일괄 예측 결과")
                            st.dataframe(
                                batch_df[["하이브리드모형 예측 결과"] + (["양성 확률"] if proba is not None else []) + feature_cols[:3]],
                                use_container_width=True
                            )
                            
                            # 결과 다운로드
                            csv = batch_df.to_csv(index=False, encoding="utf-8-sig")
                            st.download_button(
                                label="예측 결과 다운로드",
                                data=csv,
                                file_name="하이브리드모형_일괄예측결과.csv",
                                mime="text/csv"
                            )

# ----------------------
# 단계 6：성능 평가（하이브리드모형 vs 단일 모형）
# ----------------------
elif st.session_state.step == 6:
    st.subheader("📈 모델 성능 평가")
    
    if st.session_state.models["regression"] is None or st.session_state.models["decision_tree"] is None:
        st.warning("먼저「모델 학습」단계를 완료하세요")
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        reg_model = st.session_state.models["regression"]
        dt_model = st.session_state.models["decision_tree"]
        reg_weight = st.session_state.models["mixed_weights"]["regression"]
        dt_weight = st.session_state.models["mixed_weights"]["decision_tree"]
        
        # 각 모델 예측 결과 계산
        if st.session_state.task == "logit":  # 분류任务 평가
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            reg_proba = reg_model.predict_proba(X_test)[:, 1]
            dt_proba = dt_model.predict_proba(X_test)[:, 1]
            mixed_proba = reg_weight * reg_proba + dt_weight * dt_proba
            mixed_pred = (mixed_proba >= 0.5).astype(int)
            
            # 분류 지표 계산
            def calc_class_metrics(y_true, y_pred, y_proba):
                acc = accuracy_score(y_true, y_pred)
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = auc(fpr, tpr)
                return {"정확도": acc, "AUC": auc_score}
            
            reg_metrics = calc_class_metrics(y_test, reg_pred, reg_proba)
            dt_metrics = calc_class_metrics(y_test, dt_pred, dt_proba)
            mixed_metrics = calc_class_metrics(y_test, mixed_pred, mixed_proba)
            
            metrics_df = pd.DataFrame({
                "모델": ["회귀 분석（로지스틱）", "의사결정나무（분류）", "하이브리드모형"],
                "정확도": [reg_metrics["정확도"], dt_metrics["정확도"], mixed_metrics["정확도"]],
                "AUC": [reg_metrics["AUC"], dt_metrics["AUC"], mixed_metrics["AUC"]]
            }).round(3)
        
        else:  # 회귀任务 평가
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            mixed_pred = reg_weight * reg_pred + dt_weight * dt_pred
            
            # 회귀 지표 계산
            def calc_reg_metrics(y_true, y_pred):
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                return {"MAE": mae, "RMSE": rmse, "R²": r2}
            
            reg_metrics = calc_reg_metrics(y_test, reg_pred)
            dt_metrics = calc_reg_metrics(y_test, dt_pred)
            mixed_metrics = calc_reg_metrics(y_test, mixed_pred)
            
            metrics_df = pd.DataFrame({
                "모델": ["회귀 분석（선형）", "의사결정나무（회귀）", "하이브리드모형"],
                "MAE": [reg_metrics["MAE"], dt_metrics["MAE"], mixed_metrics["MAE"]],
                "RMSE": [reg_metrics["RMSE"], dt_metrics["RMSE"], mixed_metrics["RMSE"]],
                "R²": [reg_metrics["R²"], dt_metrics["R²"], mixed_metrics["R²"]]
            }).round(3)
        
        # 지표 비교 표시
        st.markdown("### 모델 성능 비교")
        st.dataframe(metrics_df, use_container_width=True)
        
        # 시각화 비교
        col1, col2 = st.columns(2)
        
        # logit（분류）작업 시각화
        if st.session_state.task == "logit":
            with col1:
                st.markdown("### ROC-AUC 곡선")
                fpr_reg, tpr_reg, _ = roc_curve(y_test, reg_proba)
                fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_proba)
                fpr_mixed, tpr_mixed, _ = roc_curve(y_test, mixed_proba)
                
                fig_auc = go.Figure()
                fig_auc.add_trace(go.Scatter(x=fpr_reg, y=tpr_reg, name=f"회귀 분석 (AUC={reg_metrics['AUC']:.3f})"))
                fig_auc.add_trace(go.Scatter(x=fpr_dt, y=tpr_dt, name=f"의사결정나무 (AUC={dt_metrics['AUC']:.3f})"))
                fig_auc.add_trace(go.Scatter(x=fpr_mixed, y=tpr_mixed, name=f"하이브리드모형 (AUC={mixed_metrics['AUC']:.3f})", line_dash="dash", line_width=3))
                fig_auc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="랜덤 추측", line_color="gray", line_dash="dot"))
                st.plotly_chart(fig_auc, use_container_width=True)
            
            with col2:
                st.markdown("### 혼동 행렬（하이브리드모형）")
                cm = confusion_matrix(y_test, mixed_pred)
                cm_df = pd.DataFrame(cm, index=["실제 음성", "실제 양성"], columns=["예측 음성", "예측 양성"])
                fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
                st.plotly_chart(fig_cm, use_container_width=True)
        
        # 의사결정나무（회귀）작업 시각화
        else:
            with col1:
                st.markdown("### 예측값 vs 실제값（하이브리드모형）")
                fig_pred = px.scatter(x=y_test, y=mixed_pred, title="실제값 vs 예측값", labels={"x": "실제값", "y": "예측값"})
                fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], line_color="red", name="이상적인 피팅 라인"))
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                st.markdown("### 잔차 그래프（하이브리드모형）")
                residuals = y_test - mixed_pred
                fig_res = px.scatter(x=mixed_pred, y=residuals, title="예측값 vs 잔차", labels={"x": "예측값", "y": "잔차"})
                fig_res.add_trace(go.Scatter(x=[mixed_pred.min(), mixed_pred.max()], y=[0, 0], line_color="red", name="잔차=0 라인"))
                st.plotly_chart(fig_res, use_container_width=True)
        
        # 모델 해석（특징 중요도：의사결정나무 기반）
        st.divider()
        st.markdown("### 모델 해석：핵심 특징 중요도")
        feature_importance = pd.DataFrame({
            "특징명": st.session_state.preprocess["feature_cols"],
            "중요도": dt_model.feature_importances_  # 의사결정나무의 특징 중요도
        }).sort_values("중요도", ascending=False).head(10)
        
        fig_importance = px.bar(feature_importance, x="중요도", y="특징명", orientation="h", color="중요도", color_continuous_scale="viridis")
        st.plotly_chart(fig_importance, use_container_width=True)
