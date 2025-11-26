import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
    st.session_state.step = 0  # 0:데이터업로드 1:데이터시각화 2:데이터전처리 3:모델학습 4:예측 5:평가 (초기설정 제거됨)
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

# 단계导航 버튼
steps = ["데이터 업로드", "데이터 시각화", "데이터 전처리", "모델 학습", "모델 예측", "성능 평가"]
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i


# ----------------------
# 3. 메인 페이지：단계별 내용 표시
# ----------------------
st.title("📊 하이브리드모형 동적 배포 프레임워크")
st.markdown("**단일 원본 데이터 파일 업로드 후，시각화→전처리→학습→예측 전과정을 한 번에 완성**")
st.markdown("### 🧩 핵심 모델：회귀 분석（Regression）+ 의사결정나무（Decision Tree）")
st.divider()

# ==============================================================================
# 메인 로직 시작
# ==============================================================================

# ----------------------
#  단계 0：데이터 업로드 (기존 단계 1에서 이동)
# ----------------------
if st.session_state.step == 0:
    st.subheader("📤 데이터 업로드")
    
    tab1, tab2 = st.tabs(["📂 내 파일 업로드", "💾 서버 기본 데이터 사용"])
    
    # 인코딩 처리를 위한 내부 함수
    def load_csv_safe(file_buffer):
        # 시도할 인코딩 목록 (순서대로 시도)
        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin1']
        
        for enc in encodings:
            try:
                file_buffer.seek(0) # 파일 포인터 초기화
                df = pd.read_csv(file_buffer, encoding=enc)
                return df, enc # 성공하면 데이터와 인코딩 반환
            except UnicodeDecodeError:
                continue # 실패하면 다음 인코딩 시도
            except Exception as e:
                return None, str(e) # 기타 에러
        return None, "모든 인코딩 시도 실패"

    with tab1:
        st.markdown("지원 형식：CSV、Parquet、Excel（.xlsx/.xls）")
        uploaded_file = st.file_uploader("데이터 파일 선택", type=["csv", "parquet", "xlsx", "xls"], key="single_file")
        
        if uploaded_file:
            try:
                df = None
                # 확장자별 로드
                if uploaded_file.name.endswith('.csv'):
                    df, enc_used = load_csv_safe(uploaded_file)
                    if df is None:
                        st.error(f"❌ CSV 파일 읽기 실패: {enc_used}")
                    else:
                        st.caption(f"ℹ️ 감지된 인코딩: {enc_used}")
                        
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                if df is not None:
                    # 인덱스 초기화 (전처리 에러 방지용 필수)
                    df = df.reset_index(drop=True)
                    st.session_state.data["merged"] = df
                    st.success(f"✅ 파일 업로드 성공! ({len(df):,} 행)")
                
            except Exception as e:
                st.error(f"❌ 파일 처리 중 오류 발생: {e}")
    
    with tab2:
        DEFAULT_FILE_PATH = "combined_loan_data.csv" 
        st.info(f"💡 **기본 데이터 설명**: 대출 관련 통합 데이터 (`{DEFAULT_FILE_PATH}`)")
        
        if st.button("기본 데이터 불러오기", type="primary"):
            if os.path.exists(DEFAULT_FILE_PATH):
                # 기본 파일도 안전하게 로드 시도
                with open(DEFAULT_FILE_PATH, 'rb') as f:
                    df_default, enc_used = load_csv_safe(f)
                
                if df_default is not None:
                    st.session_state.data["merged"] = df_default.reset_index(drop=True)
                    st.success(f"✅ 기본 데이터 로드 성공! ({len(df_default):,} 행, 인코딩: {enc_used})")
                    st.rerun()
                else:
                    st.error("❌ 기본 파일을 읽을 수 없습니다 (인코딩 오류).")
            else:
                st.error(f"⚠️ 파일을 찾을 수 없습니다: {DEFAULT_FILE_PATH}")

    # 데이터 미리보기
    if st.session_state.data.get("merged") is not None:
        df_merged = st.session_state.data["merged"]
        st.divider()
        st.markdown(f"### ✅ 현재 로드된 데이터 ({len(df_merged):,} 행)")
        st.dataframe(df_merged.head(5), width='stretch')

# ----------------------
#  단계 1：데이터 시각화 (기존 단계 2에서 이동)
# ----------------------
elif st.session_state.step == 1:
    st.subheader("📊 데이터 시각화")
    
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요")
    else:
        df = st.session_state.data["merged"]
        
        # --- 변수 선택 (Variable Selection) ---
        st.markdown("### 1️⃣ 시각화할 변수 선택")
        all_cols = df.columns.tolist()
        default_selection = all_cols[:10] if len(all_cols) > 10 else all_cols
        
        selected_cols = st.multiselect(
            "분석 대상 변수 선택",
            options=all_cols,
            default=default_selection
        )
        
        if not selected_cols:
            st.error("⚠️ 최소 하나 이상의 변수를 선택해야 시각화가 가능합니다.")
        else:
            df_vis = df[selected_cols]
            st.divider()
            
            # --- 그래프 설정 ---
            st.markdown("### 2️⃣ 그래프 설정")
            cat_cols = df_vis.select_dtypes(include=["object", "category"]).columns.tolist()
            num_cols = df_vis.select_dtypes(include=["int64", "float64"]).columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("📋 X축 (범주형)", ["선택 안 함"] + cat_cols)
                if x_var == "선택 안 함": x_var = None
            with col2:
                y_var = st.selectbox("📈 Y축 (수치형)", num_cols if num_cols else ["없음"])
            with col3:
                graph_type = st.selectbox("📊 그래프 유형", [
                    "막대 그래프", "박스 플롯", "산점도", "히스토그램", "선 그래프"
                ])
            
            st.divider()
            
            # 시각화 출력
            if y_var and y_var != "없음":
                try:
                    if graph_type == "히스토그램":
                        fig = px.histogram(df_vis, x=y_var, color=x_var, title=f"{y_var} 분포")
                    elif graph_type == "막대 그래프" and x_var:
                        avg_df = df_vis.groupby(x_var)[y_var].mean().reset_index()
                        fig = px.bar(avg_df, x=x_var, y=y_var, color=x_var, title=f"{x_var}별 {y_var} 평균")
                    elif graph_type == "박스 플롯" and x_var:
                        fig = px.box(df_vis, x=x_var, y=y_var, color=x_var, title=f"{x_var}별 {y_var} 분포")
                    elif graph_type == "산점도" and x_var:
                        fig = px.scatter(df_vis, x=x_var, y=y_var, color=x_var, title=f"{x_var} vs {y_var}")
                    elif graph_type == "선 그래프" and x_var:
                        line_df = df_vis.groupby(x_var)[y_var].mean().reset_index()
                        fig = px.line(line_df, x=x_var, y=y_var, markers=True, title=f"{x_var}별 {y_var} 추세")
                    else:
                        fig = None
                        st.info("X축 변수를 선택해주세요.")
                        
                    if fig:
                        st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"그래프 생성 오류: {e}")
            else:
                st.info("Y축 변수를 선택하면 그래프가 표시됩니다.")

# ----------------------
#  단계 2：데이터 전처리 (기존 단계 3에서 이동)
# ----------------------
elif st.session_state.step == 2:
    st.subheader("🧹 데이터 전처리 & 변수 선택")
    
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요.")
    else:
        # 원본 데이터 로드
        df_origin = st.session_state.data["merged"].copy()
        all_cols = df_origin.columns.tolist()

        st.markdown("### 1️⃣ 분석 변수 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("🎯 타겟 변수 (Y)", options=all_cols)
        
        feature_candidates = [c for c in all_cols if c != target_col]
        
        with col2:
            default_feats = feature_candidates[:10] if len(feature_candidates) > 10 else feature_candidates
            selected_features = st.multiselect(
                "📋 입력 변수 (X)",
                options=feature_candidates,
                default=default_feats
            )
        
        st.divider()

        if not selected_features:
            st.error("⚠️ 분석할 변수를 선택해주세요.")
        else:
            # 설정 저장
            st.session_state.preprocess["target_col"] = target_col
            
            tab1, tab2 = st.tabs(["⚡ 전처리 실행", "📊 중요도 분석"])
            
            with tab1:
                st.write(f"**Y(타겟) 결측치 제거** 및 **X(입력) 결측치 채우기**를 수행합니다.")
                
                if st.button("🚀 전처리 및 정제 시작", type="primary"):
                    with st.spinner("데이터 정제 중..."):
                        try:
                            # [핵심 1] 타겟(Y)이 비어있는 행 제거 (이게 없으면 NaN 에러 발생)
                            clean_df = df_origin.dropna(subset=[target_col]).reset_index(drop=True)
                            
                            dropped_count = len(df_origin) - len(clean_df)
                            if dropped_count > 0:
                                st.warning(f"⚠️ 타겟 변수({target_col})값이 비어있는 {dropped_count}개 행을 제거했습니다.")
                            
                            # X, y 분리
                            X = clean_df[selected_features].copy()
                            y = clean_df[target_col].copy()
                            
                            # [핵심 2] 타겟(Y) 데이터 인코딩 (문자일 경우 숫자로 변환)
                            # 회귀인데 Y가 문자면 에러, 분류면 자동 인코딩
                            le_target = None
                            if st.session_state.task == "logit" and y.dtype == 'object':
                                le_target = LabelEncoder()
                                y = pd.Series(le_target.fit_transform(y), index=y.index)
                                st.info("ℹ️ 타겟 변수가 문자열이라 자동으로 숫자로 변환(Encoding)되었습니다.")
                            
                            # X 데이터 전처리 시작
                            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                            
                            # 1. 값이 없는(All-NaN) 수치형 컬럼 제외
                            valid_num_cols = [c for c in num_cols if X[c].notna().sum() > 0]
                            num_cols = valid_num_cols 

                            # 변환기 준비
                            imputer = SimpleImputer(strategy='mean')
                            scaler = StandardScaler()
                            encoders = {}

                            # 2. 수치형 처리
                            if num_cols:
                                # DataFrame 할당 시 index=X.index 필수
                                X_imputed = imputer.fit_transform(X[num_cols])
                                X_scaled = scaler.fit_transform(X_imputed)
                                X[num_cols] = pd.DataFrame(X_scaled, columns=num_cols, index=X.index)
                            
                            # 3. 범주형 처리
                            for col in cat_cols:
                                X[col] = X[col].fillna("Unknown").astype(str)
                                le = LabelEncoder()
                                trans = le.fit_transform(X[col])
                                X[col] = pd.Series(trans, index=X.index)
                                encoders[col] = le
                            
                            # 최종 컬럼 정리
                            final_features = num_cols + cat_cols
                            X = X[final_features]
                            
                            # 新增：检查并处理 X 中的无穷值
                            X = X.replace([np.inf, -np.inf], np.nan)
                            
                            # 检查剩余 NaN 并报告
                            nan_counts = X.isna().sum()
                            total_nans = nan_counts.sum()
                            if total_nans > 0:
                                st.info(f"ℹ️ 입력 변수에 {total_nans}개의 결측치가 발견되어 처리됩니다.")
                            
                            # 最终检查：确保没有 NaN 残留
                            if X.isna().sum().sum() > 0:
                                st.warning("⚠️ 일부 결측치 처리에 실패했습니다. 추가 정제가 필요합니다.")
                            
                            #  전역 상태 저장
                            st.session_state.preprocess.update({
                                "feature_cols": final_features,
                                "imputer": imputer if num_cols else None,
                                "scaler": scaler if num_cols else None,
                                "encoders": encoders,
                                "num_cols": num_cols,
                                "cat_cols": cat_cols,
                                "target_encoder": le_target # Y 인코더도 저장
                            })
                            
                            # 处理된 데이터 저장
                            st.session_state.data["X_processed"] = X
                            st.session_state.data["y_processed"] = y
                            
                            st.success(f"✅ 전처리 완료! (데이터 수: {len(X)}행)")
                            st.dataframe(X.head(), width='stretch')
                            
                        except Exception as e:
                            st.error(f"❌ 오류 발생: {str(e)}")
                            
            with tab2:
                if "X_processed" in st.session_state.data and st.session_state.data["X_processed"] is not None:
                    if st.button("🔍 변수 중요도 확인"):
                        #  저장된 처리 데이터 가져오기
                        X_p = st.session_state.data["X_processed"]
                        y_p = st.session_state.data["y_processed"]
                        
                        # NaN 체크 (디버깅용)
                        if X_p.isna().sum().sum() > 0 or y_p.isna().sum() > 0:
                            st.error("❌ 데이터에 여전히 결측치(NaN)가 남아있습니다. [전처리 실행] 버튼을 다시 눌러주세요.")
                        else:
                            try:
                                # 模型 피팅
                                if st.session_state.task == "logit":
                                    model = DecisionTreeClassifier(max_depth=5, random_state=42)
                                else:
                                    model = DecisionTreeRegressor(max_depth=5, random_state=42)
                                
                                model.fit(X_p, y_p)
                                
                                imp = pd.DataFrame({
                                    "Feature": X_p.columns,
                                    "Importance": model.feature_importances_
                                }).sort_values("Importance", ascending=False)
                                
                                st.plotly_chart(
                                    px.bar(imp, x="Importance", y="Feature", orientation='h', title="변수 중요도"),
                                    width='stretch'
                                )
                            except Exception as e:
                                st.error(f"분석 실패: {e}")
                                st.warning("타겟 변수(Y)의 데이터 타입을 확인해주세요 (회귀인데 문자가 들어있는지 등).")
                else:
                    st.info("👈 먼저 [전처리 실행] 버튼을 눌러주세요.")
                    
# ==============================================================================
#  단계 3：모델 학습 (핵심 수정 부분)
# ==============================================================================
elif st.session_state.step == 3:
    st.subheader("🚀 모델 학습 설정")
    
    if "X_processed" not in st.session_state.data:
        st.warning("⚠️ 먼저 [데이터 전처리] 단계를 완료하세요.")
    else:
        # -------------------------------------------------------------
        # 1. 분석 유형 선택 (분류 vs 회귀)
        # -------------------------------------------------------------
        st.markdown("#### 1️⃣ 분석 유형 선택")
        task_option = st.radio(
            "데이터의 타겟(Y) 특성에 맞는 유형을 선택하세요:",
            ["분류 (Classification) - 예: 합격/불합격, 0/1", 
             "회귀 (Regression) - 예: 가격, 점수, 수치 예측"],
            horizontal=True
        )
        
        # task 상태 업데이트
        if "분류" in task_option:
            st.session_state.task = "logit"
        else:
            st.session_state.task = "tree" # 내부 로직상 tree/regression 구분을 위해 사용

        st.divider()

        # -------------------------------------------------------------
        # 2. 모델 전략 선택 (Decision Tree / Logic / Hybrid)
        # -------------------------------------------------------------
        st.markdown("#### 2️⃣ 사용할 모델 전략 선택")
        
        # 3가지 옵션을 탭으로 구현
        model_tabs = st.tabs(["🌲 Decision Tree (의사결정나무)", "📈 Logic (로지스틱/선형회귀)", "⚖️ Hybrid (하이브리드 모형)"])
        
        selected_strategy = None
        current_reg_weight = 0.5 # 초기값

        with model_tabs[0]:
            st.caption("의사결정나무 모델만 사용하여 예측합니다. (해석 용이, 비선형 관계 파악)")
            if st.checkbox("Decision Tree 선택", key="sel_dt"):
                selected_strategy = "dt"
                current_reg_weight = 0.0 # 회귀 비중 0 -> 트리 100%

        with model_tabs[1]:
            st.caption("로지스틱(분류) 또는 선형(회귀) 모델만 사용하여 예측합니다. (변수 영향력 파악 용이)")
            if st.checkbox("Logic(회귀/로지스틱) 선택", key="sel_reg"):
                selected_strategy = "reg"
                current_reg_weight = 1.0 # 회귀 비중 100%

        with model_tabs[2]:
            st.caption("두 모델을 결합하여 예측 성능을 극대화합니다.")
            if st.checkbox("Hybrid 모형 선택", value=True, key="sel_hybrid"): # 기본값
                selected_strategy = "hybrid"
                st.markdown("---")
                st.write("**Hybrid 가중치 설정**")
                current_reg_weight = st.slider(
                    "Logic(회귀) 모델 반영 비율", 
                    min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                    help="값이 높을수록 로지스틱/선형회귀의 영향력이 커집니다."
                )
                st.write(f"👉 **Logic: {current_reg_weight * 100:.0f}%** +  **Tree: {(1-current_reg_weight) * 100:.0f}%**")

        st.divider()

        # -------------------------------------------------------------
        # 3. 데이터 분할 및 학습 실행
        # -------------------------------------------------------------
        st.markdown("#### 3️⃣ 학습 실행")
        test_size = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.2)
        
        if st.button("🏁 모델 학습 시작", type="primary"):
            with st.spinner("모델을 학습 중입니다..."):
                try:
                    X = st.session_state.data["X_processed"]
                    y = st.session_state.data["y_processed"]
                    
                    # 데이터 분할
                    stratify_opt = y if st.session_state.task == "logit" and y.nunique() > 1 else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=stratify_opt
                    )
                    
                    # 모델 초기화
                    if st.session_state.task == "logit":
                        reg_model = LogisticRegression(max_iter=1000)
                        dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
                    else:
                        reg_model = LinearRegression()
                        dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
                    
                    # 학습 수행 (항상 둘 다 학습해두고, 예측 시 가중치로 조절하는 방식이 안전함)
                    reg_model.fit(X_train, y_train)
                    dt_model.fit(X_train, y_train)
                    
                    # 결과 저장
                    st.session_state.models["regression"] = reg_model
                    st.session_state.models["decision_tree"] = dt_model
                    
                    # [핵심] 선택한 전략에 따라 가중치 저장
                    # Decision Tree 탭 선택 -> reg=0
                    # Logic 탭 선택 -> reg=1
                    # Hybrid 탭 선택 -> reg=slider값
                    
                    # UI의 Checkbox 로직이 배타적이지 않을 수 있어, 마지막 선택 기준으로 우선순위 정리
                    # (실제 앱에서는 st.radio를 쓰는게 더 깔끔하지만, 탭 안에 넣기 위해 로직 처리)
                    final_reg_weight = 0.5 # Default Hybrid
                    
                    if selected_strategy == "dt":
                        final_reg_weight = 0.0
                        st.info("ℹ️ **Decision Tree 단일 모델**로 설정되었습니다.")
                    elif selected_strategy == "reg":
                        final_reg_weight = 1.0
                        st.info("ℹ️ **Logic (회귀/로지스틱) 단일 모델**로 설정되었습니다.")
                    elif selected_strategy == "hybrid":
                        final_reg_weight = current_reg_weight
                        st.info(f"ℹ️ **Hybrid 모델 (Logic {final_reg_weight:.1f} : Tree {1-final_reg_weight:.1f})**로 설정되었습니다.")
                    
                    st.session_state.models["mixed_weights"] = {
                        "regression": final_reg_weight,
                        "decision_tree": 1.0 - final_reg_weight
                    }
                    
                    st.session_state.data.update({"X_test": X_test, "y_test": y_test})
                    st.success("✅ 학습 완료!")
                    
                except Exception as e:
                    st.error(f"학습 중 오류 발생: {e}")

# ----------------------
#  단계 4：모델 예측 (기존 단계 5에서 이동)
# ----------------------
elif st.session_state.step == 4:
    st.subheader("🎯 모델 예측")
    
    if st.session_state.models["regression"] is None:
        st.warning("먼저「모델 학습」단계를 완료하세요")
    else:
        def predict_pipeline(input_df):
            # 1. 전처리 적용
            preprocess = st.session_state.preprocess
            X = input_df.copy()
            
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            
            #  수치형 변환
            if preprocess["imputer"]:
                X[num_cols] = preprocess["imputer"].transform(X[num_cols])
                X[num_cols] = preprocess["scaler"].transform(X[num_cols])
            
            #  범주형 변환
            for col in cat_cols:
                X[col] = X[col].fillna("알 수 없음").astype(str)
                encoder = preprocess["encoders"].get(col)
                if encoder:
                    if isinstance(encoder, LabelEncoder):
                        #  미지의 값 처리
                        known_classes = set(encoder.classes_)
                        X[col] = X[col].apply(lambda x: x if x in known_classes else "알 수 없음")
                        #  "알 수 없음"이 클래스에 없으면 추가 (임시 처리)
                        if "알 수 없음" not in known_classes:
                             #  LabelEncoder는 동적 추가가 어려우므로 0으로 대체하거나 예외처리 필요
                             #  여기서는 편의상 가장 빈도 높은 값으로 대체 가정 또는 0
                             pass 
                        #  transform 시 에러 방지를 위해 try-except 권장
                        try:
                            X[col] = encoder.transform(X[col])
                        except:
                            X[col] = 0
                    else:
                        #  OneHotEncoder
                        ohe, ohe_cols = encoder
                        ohe_result = ohe.transform(X[[col]])
                        X = pd.concat([X.drop(col, axis=1), pd.DataFrame(ohe_result, columns=ohe_cols)], axis=1)
            
            #  컬럼 순서 맞추기
            missing_cols = set(preprocess["feature_cols"]) - set(X.columns)
            for c in missing_cols:
                X[c] = 0
            X = X[preprocess["feature_cols"]]
            
            # 2. 예측
            reg_model = st.session_state.models["regression"]
            dt_model = st.session_state.models["decision_tree"]
            weights = st.session_state.models["mixed_weights"]
            
            if st.session_state.task == "logit":
                reg_p = reg_model.predict_proba(X)[:, 1]
                dt_p = dt_model.predict_proba(X)[:, 1]
                mixed_p = weights["regression"] * reg_p + weights["decision_tree"] * dt_p
                pred = (mixed_p >= 0.5).astype(int)
                return pred, mixed_p
            else:
                reg_p = reg_model.predict(X)
                dt_p = dt_model.predict(X)
                mixed_p = weights["regression"] * reg_p + weights["decision_tree"] * dt_p
                return mixed_p, None

        mode = st.radio("예측 방식", ["단일 데이터 입력", "일괄 업로드 (CSV)"])
        
        if mode == "단일 데이터 입력":
            st.markdown("#### 데이터 입력")
            feature_cols = st.session_state.preprocess["feature_cols"]
            #  원본 데이터프레임 구조 참조 (인코딩 전)
            original_features = [c for c in st.session_state.data["merged"].columns 
                               if c not in [st.session_state.preprocess["target_col"]]]
            
            input_data = {}
            with st.form("pred_form"):
                cols = st.columns(3)
                for i, col in enumerate(original_features[:9]): #  최대 9개만 표시
                    with cols[i % 3]:
                        #  원본 데이터 타입 확인
                        col_type = st.session_state.data["merged"][col].dtype
                        if pd.api.types.is_numeric_dtype(col_type):
                            input_data[col] = st.number_input(col, value=0.0)
                        else:
                            opts = st.session_state.data["merged"][col].dropna().unique()
                            input_data[col] = st.selectbox(col, options=opts)
                submit = st.form_submit_button("예측하기")
            
            if submit:
                input_df = pd.DataFrame([input_data])
                pred, proba = predict_pipeline(input_df)
                st.divider()
                if st.session_state.task == "logit":
                    st.metric("예측 결과", "양성(Positive)" if pred[0]==1 else "음성(Negative)")
                    st.metric("확률", f"{proba[0]:.2%}")
                else:
                    st.metric("예측 값", f"{pred[0]:.4f}")
                    
        else:
            up_file = st.file_uploader("CSV 업로드", type=["csv"])
            if up_file:
                batch_df = pd.read_csv(up_file)
                if st.button("일괄 예측 시작"):
                    pred, proba = predict_pipeline(batch_df)
                    batch_df["Predicted"] = pred
                    if proba is not None:
                        batch_df["Probability"] = proba
                    st.dataframe(batch_df.head())
                    st.download_button("결과 다운로드", batch_df.to_csv().encode('utf-8'), "prediction.csv")

# ----------------------
#  단계 5：성능 평가 (기존 단계 6에서 이동)
# ----------------------
elif st.session_state.step == 5:
    st.subheader("📈 모델 성능 평가")
    
    if st.session_state.models["regression"] is None:
        st.warning("먼저「모델 학습」단계를 완료하세요")
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        
        reg_model = st.session_state.models["regression"]
        dt_model = st.session_state.models["decision_tree"]
        weights = st.session_state.models["mixed_weights"]
        
        if st.session_state.task == "logit":
            #  확률 계산
            reg_p = reg_model.predict_proba(X_test)[:, 1]
            dt_p = dt_model.predict_proba(X_test)[:, 1]
            mixed_p = weights["regression"] * reg_p + weights["decision_tree"] * dt_p
            
            #  예측값
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            mixed_pred = (mixed_p >= 0.5).astype(int)
            
            #  평가 함수
            def get_metrics(y, pred, proba):
                return {
                    "ACC": accuracy_score(y, pred),
                    "AUC": auc(*roc_curve(y, proba)[:2])
                }
            
            m1 = get_metrics(y_test, reg_pred, reg_p)
            m2 = get_metrics(y_test, dt_pred, dt_p)
            m3 = get_metrics(y_test, mixed_pred, mixed_p)
            
            metrics = pd.DataFrame([m1, m2, m3], index=["회귀분석", "의사결정나무", "하이브리드"])
            st.table(metrics)
            
            #  ROC 곡선
            fpr, tpr, _ = roc_curve(y_test, mixed_p)
            fig = px.area(x=fpr, y=tpr, title=f"ROC Curve (Hybrid AUC={m3['AUC']:.3f})", 
                        labels=dict(x="False Positive Rate", y="True Positive Rate"))
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig, width='stretch')
            
        else:
            #  회귀 평가
            reg_pred = reg_model.predict(X_test)
            dt_pred = dt_model.predict(X_test)
            mixed_pred = weights["regression"] * reg_pred + weights["decision_tree"] * dt_pred
            
            def get_reg_metrics(y, pred):
                return {
                    "MAE": mean_absolute_error(y, pred),
                    "RMSE": np.sqrt(mean_squared_error(y, pred)),
                    "R2": r2_score(y, pred)
                }
            
            m1 = get_reg_metrics(y_test, reg_pred)
            m2 = get_reg_metrics(y_test, dt_pred)
            m3 = get_reg_metrics(y_test, mixed_pred)
            
            metrics = pd.DataFrame([m1, m2, m3], index=["선형회귀", "의사결정나무", "하이브리드"])
            st.table(metrics)
            
            #  예측 vs 실제
            fig = px.scatter(x=y_test, y=mixed_pred, title="실제값 vs 예측값 (Hybrid)", 
                           labels={"x": "실제값", "y": "예측값"})
            fig.add_shape(type='line', line=dict(dash='dash', color='red'), 
                        x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max())
            st.plotly_chart(fig, width='stretch')
            
        #  중요도 (Tree 기준)
        if hasattr(dt_model, "feature_importances_"):
            st.markdown("### 🌳 변수 중요도 (의사결정나무 기준)")
            imp_df = pd.DataFrame({
                "Feature": st.session_state.preprocess["feature_cols"],
                "Importance": dt_model.feature_importances_
            }).sort_values("Importance", ascending=False).head(10)
            
            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', title="Top 10 Feature Importance")
            st.plotly_chart(fig_imp, width='stretch')
