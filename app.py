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
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, mean_absolute_error, 
    mean_squared_error, r2_score
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
steps = ["데이터 업로드", "데이터 시각화", "데이터 전처리", "모델 학습", "성능 평가"]
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i


# ----------------------
# 3. 메인 페이지：단계별 내용 표시
# ----------------------
st.title("📊 하이브리드모형 동적 배포 프레임워크")
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
        DEFAULT_FILE_PATH = "Accepted_data (1).csv" 
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
        
        # ---------------------------------------------------------
        # [핵심 1] 타겟 변수(Y) 정의
        # ---------------------------------------------------------
        with col1:
            if "Loan_status" in all_cols:
                default_index = all_cols.index("Loan_status")
            else:
                default_index = 0
                
            target_col = st.selectbox(
                "🎯 타겟 변수 (Y) 선택", 
                options=all_cols,
                index=default_index,
                help="예측하고자 하는 목표 변수입니다."
            )
            
        # ---------------------------------------------------------
        # [핵심 2] 타겟 변수 정의에 따른 입력 변수(X) 후보 목록 구성
        # 타겟 변수와 입력 변수가 겹치지 않도록 리스트에서 제외합니다.
        # ---------------------------------------------------------
        feature_candidates = [c for c in all_cols if c != target_col]
        
        with col2:
            default_feats = feature_candidates[:10] if len(feature_candidates) > 10 else feature_candidates
            selected_features = st.multiselect(
                "📋 입력 변수 (X) 선택",
                options=feature_candidates, # 타겟이 제외된 리스트 사용
                default=default_feats,
                help="타겟 변수를 예측하기 위해 사용할 데이터입니다."
            )
        
        st.divider()

        if not selected_features:
            st.error("⚠️ 분석할 변수를 선택해주세요.")
        else:
            # 설정 저장
            st.session_state.preprocess["target_col"] = target_col
            
            # 탭 생성 (리스트 인덱싱으로 안전하게 접근)
            tabs = st.tabs(["⚡ 전처리 실행"])
            tab1 = tabs[0]
            
            with tab1:
                st.write(f"**Y(타겟) 결측치 제거** 및 **X(입력) 결측치 채우기**를 수행합니다.")
                
                if st.button("🚀 전처리 및 정제 시작", type="primary"):
                    with st.spinner("데이터 정제 중..."):
                        try:
                            # -----------------------------------------------------
                            # [안전 장치] 혹시라도 입력 변수에 타겟이 포함되어 있는지 재확인
                            # -----------------------------------------------------
                            if target_col in selected_features:
                                selected_features.remove(target_col)
                                st.warning(f"⚠️ 입력 변수 목록에서 타겟 변수 '{target_col}'를 자동으로 제외했습니다.")

                            # 1. 타겟(Y) 결측치 처리 (타겟이 없으면 학습 불가하므로 제거)
                            clean_df = df_origin.dropna(subset=[target_col]).reset_index(drop=True)
                            
                            dropped_count = len(df_origin) - len(clean_df)
                            if dropped_count > 0:
                                st.warning(f"⚠️ 타겟 변수({target_col})가 비어있는 {dropped_count}개 행을 제거했습니다.")
                            
                            # 데이터 분리
                            X = clean_df[selected_features].copy()
                            y = clean_df[target_col].copy()
                            
                            # -----------------------------------------------------
                            # [핵심 3] 타겟 변수(Y)의 타입에 따른 인코딩 처리
                            # 분류 문제인데 타겟이 문자열이면 LabelEncoding 수행
                            # -----------------------------------------------------
                            le_target = None
                            
                            # 로직: Task가 분류(logit)이거나, 데이터 타입이 객체(문자)인 경우
                            if y.dtype == 'object' or y.dtype.name == 'category':
                                try:
                                    le_target = LabelEncoder()
                                    y = pd.Series(le_target.fit_transform(y), index=y.index)
                                    st.info(f"ℹ️ 타겟 변수 '{target_col}'가 문자열 형식이어서 숫자로 변환(Label Encoding)했습니다.")
                                    # 인코딩 클래스 정보 표시 (예: 0=Fail, 1=Pass)
                                    mapping_info = {i: label for i, label in enumerate(le_target.classes_)}
                                    st.caption(f"└ 변환 정보: {mapping_info}")
                                except Exception as e:
                                    st.warning(f"타겟 변수 인코딩 중 이슈 발생: {e}")

                            # -----------------------------------------------------
                            # 입력 변수(X) 전처리 시작
                            # -----------------------------------------------------
                            num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                            
                            # 1. 값이 하나도 없는 컬럼 제외
                            valid_num_cols = [c for c in num_cols if X[c].notna().sum() > 0]
                            num_cols = valid_num_cols 

                            # 변환기 준비
                            imputer = SimpleImputer(strategy='mean')
                            scaler = StandardScaler()
                            encoders = {}

                            # 2. 수치형 변수 처리 (결측치 평균 대치 -> 스케일링)
                            if num_cols:
                                X_imputed = imputer.fit_transform(X[num_cols])
                                X_scaled = scaler.fit_transform(X_imputed)
                                X[num_cols] = pd.DataFrame(X_scaled, columns=num_cols, index=X.index)
                            
                            # 3. 범주형 변수 처리 (결측치 'Unknown' -> Label Encoding)
                            for col in cat_cols:
                                X[col] = X[col].fillna("Unknown").astype(str)
                                le = LabelEncoder()
                                trans = le.fit_transform(X[col])
                                X[col] = pd.Series(trans, index=X.index)
                                encoders[col] = le
                            
                            # 최종 데이터 병합 및 정리
                            final_features = num_cols + cat_cols
                            X = X[final_features]
                            X = X.replace([np.inf, -np.inf], np.nan) # 무한대 처리
                            
                            # 잔여 결측치 확인 (있으면 0으로 채움)
                            if X.isna().sum().sum() > 0:
                                st.info("ℹ️ 처리되지 않은 잔여 결측치를 0으로 대치합니다.")
                                X = X.fillna(0)
                            
                            # -----------------------------------------------------
                            # 전역 상태(Session State)에 저장
                            # -----------------------------------------------------
                            st.session_state.preprocess.update({
                                "feature_cols": final_features,
                                "imputer": imputer if num_cols else None,
                                "scaler": scaler if num_cols else None,
                                "encoders": encoders,
                                "target_encoder": le_target
                            })
                            
                            st.session_state.data["X_processed"] = X
                            st.session_state.data["y_processed"] = y
                            
                            st.markdown("### 📊 클래스 불균형 확인")
                            y_counts = y.value_counts(normalize=True)
                            st.write(y_counts)
                            
                            st.session_state.use_smote = y_counts.max() > 0.6

                            # 🔧 NEW: use_smote 初始设定（防止以后出现无定义错误
                            if "use_smote" not in st.session_state:
                                st.session_state.use_smote = False

                            
                            
                        except Exception as e:
                            st.error(f"❌ 전처리 중 오류 발생: {str(e)}")
                else:
                    st.info("👈 위 버튼을 눌러 전처리를 시작하세요.")

# ==============================================================================
#  단계 3：🚀 모델 학습 (SMOTE + class_weight + 기존 3모델 유지)
# ==============================================================================
elif st.session_state.step == 3:
    st.subheader("🚀 모델 학습 설정")

    if "X_processed" not in st.session_state.data:
        st.warning("⚠ 먼저 전처리 단계를 완료하세요.")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]

        # -------------------------------------------------------------
        # 1️⃣ 분석 유형 선택
        # -------------------------------------------------------------
        task_option = st.radio(
            "분석 유형을 선택하세요:",
            ["분류 (Classification)", "회귀 (Regression)"],
            horizontal=True
        )
        is_classification = "분류" in task_option

        st.divider()

        # -------------------------------------------------------------
        # 2️⃣ 모델 설정 & 데이터 분리 (로그/트리/하이브리드 모두 유지!)
        # -------------------------------------------------------------
        st.markdown("### 2️⃣ Logit / Tree / Hybrid 모델 설정")

        col1, col2, col3 = st.columns(3)

        # 🔹 Logit Model
        with col1:
            st.subheader("🔹 Logit")
            test_size_logit = st.slider("📌 Test 비율", 0.1, 0.4, 0.2, key="logit_test")

        # 🌳 Tree Model
        with col2:
            st.subheader("🌳 Tree")
            tree_depth = st.slider("📌 트리 깊이 (max_depth)", 2, 20, 6)
            test_size_tree = st.slider("📌 Test 비율", 0.1, 0.4, 0.2, key="tree_test")

        # ⚖️ Hybrid Model
        with col3:
            st.subheader("⚖ Hybrid")
            reg_weight = st.slider(
                "Logit 가중치",
                0.0, 1.0, 0.5, 0.1,
                key="hybrid_weight"
            )
            st.caption(f"Logit {reg_weight*100:.0f}% + Tree {(1-reg_weight)*100:.0f}%")

        st.divider()

        # -------------------------------------------------------------
        # 3️⃣ 모델 학습 시작 버튼
        # -------------------------------------------------------------
        if st.button("🏁 모델 학습 시작"):
            try:
                stratify_opt = y if is_classification else None

                # ------------------------------
                # 📌 데이터 분리 (Logit / Tree / Hybrid 각각 수행)
                # ------------------------------
                X_train_logit, X_test_logit, y_train_logit, y_test_logit = train_test_split(
                    X, y, test_size=test_size_logit, random_state=42, stratify=stratify_opt
                )
                X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
                    X, y, test_size=test_size_tree, random_state=42, stratify=stratify_opt
                )
                X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=stratify_opt
                )

                # -------------------------------------------------------------
                # 🚨 클래스 불균형 자동 감지 + SMOTE 적용 (Classification만 수행)
                # -------------------------------------------------------------
                if is_classification and st.session_state.use_smote:
                    st.info("🚨 클래스 불균형 감지 → SMOTE 적용 중...")

                    from imblearn.over_sampling import SMOTE
                    sm = SMOTE(random_state=42)

                    X_train_logit, y_train_logit = sm.fit_resample(X_train_logit, y_train_logit)
                    X_train_tree, y_train_tree = sm.fit_resample(X_train_tree, y_train_tree)
                    X_train_hybrid, y_train_hybrid = sm.fit_resample(X_train_hybrid, y_train_hybrid)

                    st.success("📈 SMOTE 적용 완료! (Train 데이터만)")

                # -------------------------------------------------------------
                # class_weight 설정
                # -------------------------------------------------------------
                class_weight_opt = 'balanced' if is_classification and st.session_state.use_smote else None

                # -------------------------------------------------------------
                # 4️⃣ 모델 생성 (Logit / Tree / Hybrid 유지)
                # -------------------------------------------------------------
                if is_classification:
                    logit_model = LogisticRegression(max_iter=1000, class_weight=class_weight_opt)
                    tree_model = DecisionTreeClassifier(max_depth=tree_depth, random_state=42, class_weight=class_weight_opt)
                else:
                    logit_model = LinearRegression()
                    tree_model = DecisionTreeRegressor(max_depth=tree_depth, random_state=42)

                # -------------------------------------------------------------
                # 5️⃣ 각 모델 학습 실행
                # -------------------------------------------------------------
                logit_model.fit(X_train_logit, y_train_logit)
                tree_model.fit(X_train_tree, y_train_tree)

                # -------------------------------------------------------------
                # 6️⃣ Hybrid 가중치 저장
                # -------------------------------------------------------------
                st.session_state.models.update({
                    "logit_model": logit_model,
                    "tree_model": tree_model,
                    "hybrid_weight": reg_weight
                })

                st.session_state.data.update({
                    "X_test_logit": X_test_logit, "y_test_logit": y_test_logit,
                    "X_test_tree": X_test_tree, "y_test_tree": y_test_tree,
                    "X_test_hybrid": X_test_hybrid, "y_test_hybrid": y_test_hybrid
                })

                st.success("🎯 모든 모델 학습 완료! → 성능 평가 단계로 이동하세요.")

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

                            
# ==============================================================================
#  단계 4：성능 평가 (확장된 지표 및 혼동행렬 추가)
# ==============================================================================
elif st.session_state.step == 4:
    st.subheader("📈 모델 성능 심층 평가")
    
    if st.session_state.models["regression"] is None:
        st.warning("⚠️ 먼저 [모델 학습] 단계를 완료하세요")
    else:
        # 데이터 및 모델 로드
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        
        reg_model = st.session_state.models["regression"]
        dt_model = st.session_state.models["decision_tree"]
        w = st.session_state.models["mixed_weights"]
        
        st.info(f"ℹ️ Hybrid 가중치: Logit {w['regression']*100:.0f}% + Tree {w['decision_tree']*100:.0f}%")
        
        # ----------------------------------------------------------------------
        # A. 분류 (Classification) 평가 로직 - [요청하신 기능 집중 구현]
        # ----------------------------------------------------------------------
        if st.session_state.task == "logit":
            # 1. 확률 및 클래스 예측
            # (1) Logic
            prob_reg = reg_model.predict_proba(X_test)[:, 1]
            pred_reg = reg_model.predict(X_test)
            # (2) Tree
            prob_dt = dt_model.predict_proba(X_test)[:, 1]
            pred_dt = dt_model.predict(X_test)
            # (3) Hybrid
            prob_hybrid = (prob_reg * w["regression"]) + (prob_dt * w["decision_tree"])
            pred_hybrid = (prob_hybrid >= 0.5).astype(int)
            
            # 2. 성능 지표 계산 함수 (Accuracy, Precision, Recall, F1)
            def get_cls_detailed_metrics(y_true, y_pred, y_prob):
                return {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, zero_division=0),
                    "Recall": recall_score(y_true, y_pred, zero_division=0),
                    "F1-Score": f1_score(y_true, y_pred, zero_division=0),
                    "AUC": auc(*roc_curve(y_true, y_prob)[:2])
                }

            metrics_reg = get_cls_detailed_metrics(y_test, pred_reg, prob_reg)
            metrics_dt = get_cls_detailed_metrics(y_test, pred_dt, prob_dt)
            metrics_hybrid = get_cls_detailed_metrics(y_test, pred_hybrid, prob_hybrid)
            
            # 3. 모델별 성능 비교표 출력
            st.markdown("### 1️⃣ 모델별 주요 성능 지표")
            df_metrics = pd.DataFrame([metrics_reg, metrics_dt, metrics_hybrid], 
                                      index=["Logit Model", "Tree Model", "Hybrid Model"])
            st.table(df_metrics.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"))

            # 4. ROC Curve 비교 시각화
            st.markdown("### 2️⃣ ROC Curve 비교")
            fig_roc = go.Figure()
            def add_roc_trace(y_true, y_prob, name, color):
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name, line=dict(color=color, width=2)))

            add_roc_trace(y_test, prob_reg, "Logit", "blue")
            add_roc_trace(y_test, prob_dt, "Tree", "green")
            add_roc_trace(y_test, prob_hybrid, "Hybrid", "red")
            
            fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curves")
            st.plotly_chart(fig_roc, width='stretch')

            # 5. Confusion Matrix (혼동 행렬) 시각화
            st.markdown("### 3️⃣ Confusion Matrix (혼동 행렬)")
            st.caption("각 모델이 정답을 어떻게 맞추고 틀렸는지 시각적으로 확인합니다.")
            
            cm_col1, cm_col2, cm_col3 = st.columns(3)
            
            def plot_confusion_matrix(y_true, y_pred, title):
                cm = confusion_matrix(y_true, y_pred)
                # 히트맵 생성
                fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['0 (Neg)', '1 (Pos)'], y=['0 (Neg)', '1 (Pos)'])
                fig.update_layout(title=title, width=300, height=300, margin=dict(l=20, r=20, t=40, b=20))
                return fig

            with cm_col1:
                st.plotly_chart(plot_confusion_matrix(y_test, pred_reg, "Logit Model"), use_container_width=True)
            with cm_col2:
                st.plotly_chart(plot_confusion_matrix(y_test, pred_dt, "Tree Model"), use_container_width=True)
            with cm_col3:
                st.plotly_chart(plot_confusion_matrix(y_test, pred_hybrid, "Hybrid Model"), use_container_width=True)

        # ----------------------------------------------------------------------
        # B. 회귀 (Regression) 평가 로직
        # ----------------------------------------------------------------------
        else:
            # 1. 예측값 계산
            pred_reg = reg_model.predict(X_test)
            pred_dt = dt_model.predict(X_test)
            pred_hybrid = (pred_reg * w["regression"]) + (pred_dt * w["decision_tree"])
            
            # 2. 성능 지표
            def get_reg_metrics(y_true, y_pred):
                return {
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "R²": r2_score(y_true, y_pred)
                }
            
            m1 = get_reg_metrics(y_test, pred_reg)
            m2 = get_reg_metrics(y_test, pred_dt)
            m3 = get_reg_metrics(y_test, pred_hybrid)
            
            st.markdown("### 1️⃣ 회귀 모델 성능 지표")
            df_reg = pd.DataFrame([m1, m2, m3], index=["Logit", "Tree", "Hybrid"])
            st.table(df_reg.style.format("{:.4f}"))
            
            st.markdown("### 2️⃣ 예측값 vs 실제값 비교")
            fig = px.scatter(x=y_test, y=pred_hybrid, title="Hybrid 예측 결과", labels={'x':'실제값', 'y':'예측값'})
            fig.add_shape(type='line', line=dict(dash='dash', color='red'), x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max())
            st.plotly_chart(fig, width='stretch')
