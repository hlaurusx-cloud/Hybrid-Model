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
#  단계 2：데이터 전처리 (변수 선택보다 먼저)
# ----------------------
elif st.session_state.step == 2:
    st.subheader("🧹 데이터 전처리 & 변수 선택")

    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요.")
    else:
        df_origin = st.session_state.data["merged"].copy()
        all_cols = df_origin.columns.tolist()

        # ---------------------------------------------------------
        # 1️⃣ 전처리 먼저 실행
        # ---------------------------------------------------------
        st.markdown("### 1️⃣ 📌 데이터 전처리를 먼저 실행하세요")

        if st.button("🚀 전처리 실행", type="primary"):
            with st.spinner("데이터 전처리 중..."):

                clean_df = df_origin.copy()

                # 1. 타겟(Y) 제거 없이 전체 평균 결측치 처리
                missing_ratio = clean_df.isna().mean()
                drop_cols = missing_ratio[missing_ratio >= 0.95].index.tolist()

                if drop_cols:
                    clean_df = clean_df.drop(columns=drop_cols)
                    st.warning(f"⚠️ 결측치 95% 이상 변수 제거됨: {drop_cols}")

                # 수치형 / 범주형 나누기
                num_cols = clean_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                cat_cols = clean_df.select_dtypes(include=['object', 'category']).columns.tolist()

                # 2. 숫자형 결측치 처리 + IQR 이상치 처리 + 스케일링
                imputer = SimpleImputer(strategy='mean')
                scaler = StandardScaler()
                outlier_bounds = {}

                if num_cols:
                    X_num = clean_df[num_cols].copy()
                    X_num = pd.DataFrame(imputer.fit_transform(X_num), columns=num_cols)

                    # IQR 이상치 처리
                    for col in num_cols:
                        q1 = X_num[col].quantile(0.25)
                        q3 = X_num[col].quantile(0.75)
                        iqr = q3 - q1
                        if iqr > 0:
                            low = q1 - 1.5 * iqr
                            high = q3 + 1.5 * iqr
                            X_num[col] = X_num[col].clip(low, high)
                            outlier_bounds[col] = (low, high)

                    X_num = pd.DataFrame(scaler.fit_transform(X_num), columns=num_cols)

                    clean_df[num_cols] = X_num

                # 3. 문자형 결측치 처리 + LabelEncoding
                encoders = {}
                for col in cat_cols:
                    clean_df[col] = clean_df[col].fillna("Unknown").astype(str)
                    le = LabelEncoder()
                    clean_df[col] = le.fit_transform(clean_df[col])
                    encoders[col] = le

                clean_df = clean_df.replace([np.inf, -np.inf], np.nan).fillna(0)

                # 전처리 결과 저장
                st.session_state.data["df_processed"] = clean_df
                st.success("✅ 전처리 완료!")

                st.dataframe(clean_df.head(), use_container_width=True)

        st.divider()

        # ---------------------------------------------------------
        # 2️⃣ 전처리 된 데이터로 변수 선택
        # ---------------------------------------------------------
        if "df_processed" in st.session_state.data:
            df_processed = st.session_state.data["df_processed"]
            cols_processed = df_processed.columns.tolist()

            st.markdown("### 2️⃣ 변수 선택 (전처리된 컬럼 기준)")

            # 입력변수(X) 먼저 선택
            selected_features = st.multiselect(
                "📋 입력 변수 (X) 선택",
                options=cols_processed,
                default=cols_processed[:10]
            )

            # 타겟변수(Y) 선택
            remaining_cols = [c for c in cols_processed if c not in selected_features]

            target_col = st.selectbox(
                "🎯 타겟 변수 (Y) 선택",
                options=remaining_cols
            )

            # Y가 X에 있으면 제거
            if target_col in selected_features:
                selected_features.remove(target_col)

            # 최종 저장
            st.session_state.preprocess["target_col"] = target_col
            st.session_state.preprocess["feature_cols"] = selected_features

            st.success(f"✅ 변수 선택 완료! (X: {len(selected_features)}개 / Y: {target_col})")



# ==============================================================================
#  단계 3：🚀 모델 학습 (Logit / Tree / Hybrid)
# ==============================================================================
elif st.session_state.step == 3:
    st.subheader("🚀 모델 학습 설정")

    if "X_processed" not in st.session_state.data:
        st.warning("⚠ 먼저 전처리 단계를 완료하세요.")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]

        # -------------------------------------------------------------
        # 1️⃣ 분석 유형 선택 (분류 / 회귀)
        # -------------------------------------------------------------
        task_option = st.radio(
            "분석 유형을 선택하세요:",
            ["분류 (Classification)", "회귀 (Regression)"],
            horizontal=True
        )
        is_classification = "분류" in task_option
        st.session_state["is_classification"] = is_classification

        st.divider()

        # -------------------------------------------------------------
        # 2️⃣ 모델별 하이퍼파라미터 설정
        # -------------------------------------------------------------
        st.markdown("### 2️⃣ 모델별 하이퍼파라미터 설정")

        # 🔹 Logit / Linear Model 설정
        with st.expander("🔹 Logit (분류) / Linear (회귀) 모델 설정", expanded=True):
            test_size_logit = st.slider(
                "📌 Logit / Linear 모델용 Test 비율",
                0.1, 0.4, 0.2, key="logit_test"
            )

            if is_classification:
                # 👉 로지스틱 회귀 세부 설정
                C_logit = st.slider(
                    "🔧 Logit 규제 강도(C)",
                    0.01, 10.0, 1.0, 0.01
                )
                max_iter_logit = st.slider(
                    "🔧 Logit 최대 반복 횟수 (max_iter)",
                    100, 5000, 1000, 100
                )
                st.caption("※ solver는 'lbfgs', penalty=L2 로 고정합니다.")
            else:
                st.caption("회귀 선택 시 LinearRegression 기본 값을 사용합니다.")

        # 🌳 Tree Model 설정
        with st.expander("🌳 Tree 모델 설정 (Decision Tree)", expanded=True):
            test_size_tree = st.slider(
                "📌 Tree 모델용 Test 비율",
                0.1, 0.4, 0.2, key="tree_test"
            )
            tree_depth = st.slider(
                "🔧 트리 깊이 (max_depth)",
                2, 20, 6
            )
            # ⛔ min_samples_split, min_samples_leaf 제거

        # ⚖️ Hybrid Model 설정
        with st.expander("⚖ Hybrid 모델 설정", expanded=True):
            test_size_hybrid = st.slider(
                "📌 Hybrid 모델용 Test 비율",
                0.1, 0.4, 0.2, key="hybrid_test"
            )
            reg_weight = st.slider(
                "Logit 가중치",
                0.0, 1.0, 0.5, 0.1,
                key="hybrid_weight"
            )
            st.caption(f"👉 최종 예측 = Logit {reg_weight*100:.0f}% + Tree {(1-reg_weight)*100:.0f}%")

        st.divider()

        # -------------------------------------------------------------
        # 3️⃣ 모델 학습 시작
        # -------------------------------------------------------------
        if st.button("🏁 모델 학습 시작"):
            try:
                stratify_opt = y if is_classification else None

                # ------------------------------
                # 데이터 분리
                # ------------------------------
                X_train_logit, X_test_logit, y_train_logit, y_test_logit = train_test_split(
                    X, y, test_size=test_size_logit, random_state=42,
                    stratify=stratify_opt if is_classification else None
                )
                X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
                    X, y, test_size=test_size_tree, random_state=42,
                    stratify=stratify_opt if is_classification else None
                )
                X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = train_test_split(
                    X, y, test_size=test_size_hybrid, random_state=42,
                    stratify=stratify_opt if is_classification else None
                )

                # -------------------------------------------------------------
                # 4️⃣ 모델 생성 (Logit / Tree)
                # -------------------------------------------------------------
                if is_classification:
                    logit_model = LogisticRegression(
                        max_iter=max_iter_logit,
                        C=C_logit,
                        solver="lbfgs"
                    )
                    tree_model = DecisionTreeClassifier(
                        max_depth=tree_depth,
                        random_state=42
                    )
                else:
                    logit_model = LinearRegression()
                    tree_model = DecisionTreeRegressor(
                        max_depth=tree_depth,
                        random_state=42
                    )

                # -------------------------------------------------------------
                # 5️⃣ 모델 학습 실행
                # -------------------------------------------------------------
                logit_model.fit(X_train_logit, y_train_logit)
                tree_model.fit(X_train_tree, y_train_tree)

                # -------------------------------------------------------------
                # 6️⃣ Hybrid 저장
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

                st.success("🎯 모든 모델 학습 완료! 성능 평가 단계로 이동하세요.")

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")


# ==============================================================================
#  단계 4：성능 평가 (확장된 지표 및 혼동행렬 추가)
# ==============================================================================
elif st.session_state.step == 4:
    st.subheader("📈 모델 성능 심층 평가")

    # 1. 모델이 학습되었는지 확인
    if "logit_model" not in st.session_state.models or "tree_model" not in st.session_state.models:
        st.warning("⚠️ 먼저 [모델 학습] 단계를 완료하세요")
    else:
        # 🔹 분류 / 회귀 플래그 (step 3에서 저장한 값 사용)
        is_classification = st.session_state.get("is_classification", True)

        # ------------------------------------------------------------------
        # ✅ 2. 데이터 및 모델 로드
        # ------------------------------------------------------------------
        X_test = st.session_state.data["X_test_hybrid"]
        y_test = st.session_state.data["y_test_hybrid"]
        
        reg_model = st.session_state.models["logit_model"]     # 분류일 땐 Logit, 회귀일 땐 LinearRegression
        dt_model  = st.session_state.models["tree_model"]      # 분류일 땐 TreeClassifier, 회귀일 땐 TreeRegressor
        w         = st.session_state.models["hybrid_weight"]   # Logit 가중치 (0~1)
        
        st.info(f"ℹ️ Hybrid 가중치: Logit {w*100:.0f}% + Tree {(1-w)*100:.0f}%")
        
        # ----------------------------------------------------------------------
        # A. 분류 (Classification) 평가 로직
        # ----------------------------------------------------------------------
        if is_classification:
            # 1. 확률 및 클래스 예측
            # (1) Logit
            prob_reg = reg_model.predict_proba(X_test)[:, 1]
            pred_reg = reg_model.predict(X_test)
            
            # (2) Tree
            prob_dt = dt_model.predict_proba(X_test)[:, 1]
            pred_dt = dt_model.predict(X_test)
            
            # (3) Hybrid
            prob_hybrid = (prob_reg * w) + (prob_dt * (1 - w))
            pred_hybrid = (prob_hybrid >= 0.5).astype(int)
            
            # 2. 성능 지표 계산 함수
            def get_cls_detailed_metrics(y_true, y_pred, y_prob):
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                return {
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, zero_division=0),
                    "Recall": recall_score(y_true, y_pred, zero_division=0),
                    "F1-Score": f1_score(y_true, y_pred, zero_division=0),
                    "AUC": auc(fpr, tpr)
                }

            metrics_reg     = get_cls_detailed_metrics(y_test, pred_reg, prob_reg)
            metrics_dt      = get_cls_detailed_metrics(y_test, pred_dt, prob_dt)
            metrics_hybrid  = get_cls_detailed_metrics(y_test, pred_hybrid, prob_hybrid)
            
            # 3. 모델별 성능 비교표 출력
            st.markdown("### 1️⃣ 모델별 주요 성능 지표")
            df_metrics = pd.DataFrame(
                [metrics_reg, metrics_dt, metrics_hybrid], 
                index=["Logit Model", "Tree Model", "Hybrid Model"]
            )
            st.table(df_metrics.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"))

            # 4. ROC Curve 비교 시각화
            st.markdown("### 2️⃣ ROC Curve 비교")
            fig_roc = go.Figure()
            def add_roc_trace(y_true, y_prob, name, color):
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines', name=name,
                    line=dict(color=color, width=2)
                ))

            add_roc_trace(y_test, prob_reg,    "Logit",  "blue")
            add_roc_trace(y_test, prob_dt,     "Tree",   "green")
            add_roc_trace(y_test, prob_hybrid, "Hybrid", "red")
            
            fig_roc.add_shape(
                type='line',
                line=dict(dash='dash', color='gray'),
                x0=0, x1=1, y0=0, y1=1
            )
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                title="ROC Curves"
            )
            st.plotly_chart(fig_roc, use_container_width=True)

            # 5. Confusion Matrix (혼동 행렬) 시각화
            st.markdown("### 3️⃣ Confusion Matrix (혼동 행렬)")
            st.caption("각 모델이 정답을 어떻게 맞추고 틀렸는지 시각적으로 확인합니다.")
            
            cm_col1, cm_col2, cm_col3 = st.columns(3)
            
            def plot_confusion_matrix(y_true, y_pred, title):
                cm = confusion_matrix(y_true, y_pred)
                fig = px.imshow(
                    cm, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['0 (Neg)', '1 (Pos)'], y=['0 (Neg)', '1 (Pos)']
                )
                fig.update_layout(
                    title=title,
                    width=300, height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                return fig

            with cm_col1:
                st.plotly_chart(
                    plot_confusion_matrix(y_test, pred_reg, "Logit Model"),
                    use_container_width=True
                )
            with cm_col2:
                st.plotly_chart(
                    plot_confusion_matrix(y_test, pred_dt, "Tree Model"),
                    use_container_width=True
                )
            with cm_col3:
                st.plotly_chart(
                    plot_confusion_matrix(y_test, pred_hybrid, "Hybrid Model"),
                    use_container_width=True
                )

        # ----------------------------------------------------------------------
        # B. 회귀 (Regression) 평가 로직
        # ----------------------------------------------------------------------
        else:
            # 1. 예측값 계산
            pred_reg     = reg_model.predict(X_test)
            pred_dt      = dt_model.predict(X_test)
            pred_hybrid  = (pred_reg * w) + (pred_dt * (1 - w))
            
            # 2. 성능 지표 함수
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
            df_reg = pd.DataFrame([m1, m2, m3], index=["Linear(전 Logit 자리)", "Tree", "Hybrid"])
            st.table(df_reg.style.format("{:.4f}"))
            
            st.markdown("### 2️⃣ 예측값 vs 실제값 비교 (Hybrid)")
            fig = px.scatter(
                x=y_test, y=pred_hybrid,
                title="Hybrid 예측 결과",
                labels={'x':'실제값', 'y':'예측값'}
            )
            fig.add_shape(
                type='line',
                line=dict(dash='dash', color='red'),
                x0=y_test.min(), x1=y_test.max(),
                y0=y_test.min(), y1=y_test.max()
            )
            st.plotly_chart(fig, use_container_width=True)
