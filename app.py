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
        DEFAULT_FILE_PATH = "accepted_data.csv" 
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
            
            [tab1] = st.tabs(["⚡ 전처리 실행"])
            
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
                            

                else:
                    st.info("👈 먼저 [전처리 실행] 버튼을 눌러주세요.")
                    
# ==============================================================================
#  단계 3：모델 학습 (3개 모델 동시 학습 및 비교 준비)
# ==============================================================================
elif st.session_state.step == 3:
    st.subheader("🚀 모델 학습 (비교 분석)")
    
    if "X_processed" not in st.session_state.data:
        st.warning("⚠️ 먼저 [데이터 전처리] 단계를 완료하세요.")
    else:
        st.info("💡 **Logic(선형/로지스틱)**, **Tree(의사결정나무)**, 그리고 **Hybrid(하이브리드)** 모델을 모두 학습하여 성능을 비교합니다.")

        col1, col2 = st.columns(2)
        
        # 1. 분석 유형 선택
        with col1:
            st.markdown("#### 1️⃣ 분석 유형 선택")
            task_option = st.radio(
                "타겟(Y) 특성:",
                ["분류 (Classification)", "회귀 (Regression)"],
                horizontal=True
            )
            # task 설정
            st.session_state.task = "logit" if "분류" in task_option else "tree"

        # 2. 하이브리드 가중치 설정 (비교를 위해 정의 필요)
        with col2:
            st.markdown("#### 2️⃣ 하이브리드 가중치 설정")
            st.caption("비교 대상인 '하이브리드 모델'을 만들 때 사용할 가중치를 설정합니다.")
            reg_weight = st.slider(
                "Logic 모델 반영 비율 (나머지는 Tree)", 
                min_value=0.0, max_value=1.0, value=0.5, step=0.1
            )
            st.write(f"⚖️ Hybrid 구성: **Logic {reg_weight*100:.0f}% + Tree {(1-reg_weight)*100:.0f}%**")

        st.divider()

        # 3. 학습 실행
        st.markdown("#### 3️⃣ 전체 모델 학습 실행")
        test_size = st.slider("테스트 데이터 비율 (Test Size)", 0.1, 0.4, 0.2)
        
        if st.button("🏁 3개 모델 동시 학습 시작", type="primary"):
            with st.spinner("Logic, Tree, Hybrid 모델을 모두 학습 중입니다..."):
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
                    
                    # [핵심] 두 모델 모두 학습 (선택이 아니라 필수)
                    reg_model.fit(X_train, y_train)
                    dt_model.fit(X_train, y_train)
                    
                    # 모델 저장
                    st.session_state.models["regression"] = reg_model
                    st.session_state.models["decision_tree"] = dt_model
                    
                    # 하이브리드 계산을 위한 가중치 저장
                    st.session_state.models["mixed_weights"] = {
                        "regression": reg_weight,
                        "decision_tree": 1.0 - reg_weight
                    }
                    
                    # 테스트 데이터 저장
                    st.session_state.data.update({"X_test": X_test, "y_test": y_test})
                    
                    st.success("✅ 모든 모델 학습 완료! 다음 [성능 평가] 단계에서 비교 결과를 확인하세요.")
                    
                except Exception as e:
                    st.error(f"학습 중 오류 발생: {e}")

# ==============================================================================
#  단계 4：성능 평가 (3개 모델 비교 표 및 차트 출력)
# ==============================================================================
elif st.session_state.step == 4:
    st.subheader("📈 모델 성능 비교 평가")
    
    if st.session_state.models["regression"] is None:
        st.warning("⚠️ 먼저 [모델 학습] 단계를 완료하세요")
    else:
        # 데이터 및 모델 로드
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        
        reg_model = st.session_state.models["regression"]
        dt_model = st.session_state.models["decision_tree"]
        w = st.session_state.models["mixed_weights"]
        
        st.markdown("### 1️⃣ 모델별 성능 비교표")
        
        # ---------------------------
        # A. 분류 (Classification) 비교
        # ---------------------------
        if st.session_state.task == "logit":
            # 1. 각 모델 예측 확률 계산
            prob_reg = reg_model.predict_proba(X_test)[:, 1]
            prob_dt = dt_model.predict_proba(X_test)[:, 1]
            prob_hybrid = (prob_reg * w["regression"]) + (prob_dt * w["decision_tree"])
            
            # 2. 예측 클래스 결정 (Threshold 0.5)
            pred_reg = reg_model.predict(X_test)
            pred_dt = dt_model.predict(X_test)
            pred_hybrid = (prob_hybrid >= 0.5).astype(int)
            
            # 3. 성능 지표 계산 함수
            def calc_cls_metrics(y_true, y_pred, y_prob):
                acc = accuracy_score(y_true, y_pred)
                try:
                    auc_score = auc(*roc_curve(y_true, y_prob)[:2])
                except:
                    auc_score = 0.0
                return [acc, auc_score]

            # 4. 데이터프레임 생성
            m_reg = calc_cls_metrics(y_test, pred_reg, prob_reg)
            m_dt = calc_cls_metrics(y_test, pred_dt, prob_dt)
            m_hybrid = calc_cls_metrics(y_test, pred_hybrid, prob_hybrid)
            
            metrics_df = pd.DataFrame(
                [m_reg, m_dt, m_hybrid],
                columns=["정확도 (Accuracy)", "AUC Score"],
                index=["Logic (로지스틱)", "Tree (의사결정나무)", "Hybrid (하이브리드)"]
            )
            
            # 표 출력
            st.table(metrics_df.style.highlight_max(axis=0, color='lightgreen'))
            
            # 차트 출력 (ROC Curve 중첩)
            st.markdown("### 2️⃣ ROC Curve 비교")
            fig = go.Figure()
            
            def add_roc_trace(y_true, y_prob, name, color):
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name, line=dict(color=color)))

            add_roc_trace(y_test, prob_reg, "Logic", "blue")
            add_roc_trace(y_test, prob_dt, "Tree", "green")
            add_roc_trace(y_test, prob_hybrid, "Hybrid", "red")
            
            fig.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
            fig.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, width='stretch')

        # ---------------------------
        # B. 회귀 (Regression) 비교
        # ---------------------------
        else:
            # 1. 각 모델 예측값 계산
            pred_reg = reg_model.predict(X_test)
            pred_dt = dt_model.predict(X_test)
            pred_hybrid = (pred_reg * w["regression"]) + (pred_dt * w["decision_tree"])
            
            # 2. 성능 지표 계산 함수
            def calc_reg_metrics(y_true, y_pred):
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                return [mae, rmse, r2]
            
            # 3. 데이터프레임 생성
            m_reg = calc_reg_metrics(y_test, pred_reg)
            m_dt = calc_reg_metrics(y_test, pred_dt)
            m_hybrid = calc_reg_metrics(y_test, pred_hybrid)
            
            metrics_df = pd.DataFrame(
                [m_reg, m_dt, m_hybrid],
                columns=["MAE (평균오차)", "RMSE", "R² (결정계수)"],
                index=["Logic (선형회귀)", "Tree (의사결정나무)", "Hybrid (하이브리드)"]
            )
            
            # 표 출력 (MAE, RMSE는 낮을수록 좋음 / R2는 높을수록 좋음)
            st.table(metrics_df.style.format("{:.4f}"))
            
            # 차트 출력 (비교 막대 그래프)
            st.markdown("### 2️⃣ 성능 지표 시각화 (R² Score)")
            
            # Plotly Bar Chart
            comp_df = metrics_df.reset_index().rename(columns={"index": "Model"})
            fig = px.bar(
                comp_df, x="Model", y="R² (결정계수)", 
                color="Model", text="R² (결정계수)",
                title="모델별 R² Score 비교 (높을수록 좋음)"
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig, width='stretch')
