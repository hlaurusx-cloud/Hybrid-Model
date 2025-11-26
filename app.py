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
    page_title="하이브리드모형 동적 프레임워크",
    page_icon="📊",
    layout="wide"
)

# 전역 상태 관리
if "step" not in st.session_state:
    st.session_state.step = 0 
if "data" not in st.session_state:
    st.session_state.data = {"merged": None}
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {"imputer": None, "scaler": None, "encoders": None, "feature_cols": None, "target_col": None}
if "models" not in st.session_state:
    # mixed_weights 기본값 설정
    st.session_state.models = {"regression": None, "decision_tree": None, "mixed_weights": {"regression": 0.5, "decision_tree": 0.5}}
if "task" not in st.session_state:
    st.session_state.task = "logit" # 기본값

# ----------------------
# 2. 사이드바：단계 네비게이션
# ----------------------
st.sidebar.title("📌 단계별 진행")
st.sidebar.divider()

steps = ["데이터 업로드", "데이터 시각화", "데이터 전처리", "모델 학습", "모델 예측", "성능 평가"]
for i, step_name in enumerate(steps):
    # 버튼을 누르면 해당 단계로 이동
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i

st.sidebar.divider()
st.sidebar.info(f"현재 단계: **{steps[st.session_state.step]}**")

# ----------------------
# 3. 메인 페이지 로직
# ----------------------
st.title("📊 하이브리드모형 분석 프레임워크")
st.divider()

# ==============================================================================
#  단계 0：데이터 업로드
# ==============================================================================
if st.session_state.step == 0:
    st.subheader("📤 데이터 업로드")
    
    tab1, tab2 = st.tabs(["📂 내 파일 업로드", "💾 서버 기본 데이터 사용"])
    
    def load_csv_safe(file_buffer):
        encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig', 'latin1']
        for enc in encodings:
            try:
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer, encoding=enc)
                return df, enc
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return None, str(e)
        return None, "모든 인코딩 시도 실패"

    with tab1:
        uploaded_file = st.file_uploader("데이터 파일 선택 (CSV, Excel)", type=["csv", "parquet", "xlsx", "xls"], key="single_file")
        if uploaded_file:
            try:
                df = None
                if uploaded_file.name.endswith('.csv'):
                    df, enc_used = load_csv_safe(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                if df is not None:
                    df = df.reset_index(drop=True)
                    st.session_state.data["merged"] = df
                    st.success(f"✅ 파일 업로드 성공! ({len(df):,} 행)")
                else:
                    st.error("❌ 파일 읽기 실패")
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")
    
    with tab2:
        DEFAULT_FILE_PATH = "combined_loan_data.csv" 
        if st.button("기본 데이터 불러오기", type="primary"):
            if os.path.exists(DEFAULT_FILE_PATH):
                with open(DEFAULT_FILE_PATH, 'rb') as f:
                    df_default, enc_used = load_csv_safe(f)
                if df_default is not None:
                    st.session_state.data["merged"] = df_default.reset_index(drop=True)
                    st.success(f"✅ 기본 데이터 로드 성공! ({len(df_default):,} 행)")
                    st.rerun()
                else:
                    st.error("❌ 기본 파일 인코딩 오류")
            else:
                st.error("⚠️ 서버에 기본 파일이 없습니다.")

    if st.session_state.data.get("merged") is not None:
        st.divider()
        st.dataframe(st.session_state.data["merged"].head(), width='stretch')

# ==============================================================================
#  단계 1：데이터 시각화
# ==============================================================================
elif st.session_state.step == 1:
    st.subheader("📊 데이터 시각화")
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요")
    else:
        df = st.session_state.data["merged"]
        all_cols = df.columns.tolist()
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("#### 설정")
            x_var = st.selectbox("X축 변수", ["선택 안 함"] + all_cols)
            y_var = st.selectbox("Y축 변수", ["선택 안 함"] + all_cols)
            graph_type = st.selectbox("그래프 유형", ["막대 그래프", "박스 플롯", "산점도", "히스토그램", "선 그래프"])
        
        with col2:
            if y_var != "선택 안 함":
                try:
                    if x_var == "선택 안 함": x_var = None
                    if graph_type == "히스토그램":
                        fig = px.histogram(df, x=y_var, color=x_var, title=f"{y_var} 분포")
                    elif graph_type == "막대 그래프" and x_var:
                        avg_df = df.groupby(x_var)[y_var].mean().reset_index() if pd.api.types.is_numeric_dtype(df[y_var]) else df
                        fig = px.bar(avg_df, x=x_var, y=y_var, color=x_var)
                    elif graph_type == "박스 플롯" and x_var:
                        fig = px.box(df, x=x_var, y=y_var, color=x_var)
                    elif graph_type == "산점도" and x_var:
                        fig = px.scatter(df, x=x_var, y=y_var, color=x_var)
                    elif graph_type == "선 그래프" and x_var:
                        fig = px.line(df, x=x_var, y=y_var)
                    else:
                        fig = None
                        st.info("올바른 X, Y 변수를 선택해주세요.")
                    
                    if fig: st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"시각화 오류: {e}")

# ==============================================================================
#  단계 2：데이터 전처리
# ==============================================================================
elif st.session_state.step == 2:
    st.subheader("🧹 데이터 전처리")
    if st.session_state.data["merged"] is None:
        st.warning("⚠️ 먼저 '데이터 업로드' 단계를 완료하세요.")
    else:
        df_origin = st.session_state.data["merged"].copy()
        all_cols = df_origin.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("🎯 타겟 변수 (Y)", options=all_cols)
        with col2:
            feature_candidates = [c for c in all_cols if c != target_col]
            selected_features = st.multiselect("📋 입력 변수 (X)", options=feature_candidates, default=feature_candidates[:10])

        if st.button("🚀 전처리 실행", type="primary"):
            if not selected_features:
                st.error("입력 변수를 선택해주세요.")
            else:
                with st.spinner("데이터 정제 중..."):
                    try:
                        # 1. Y 결측 제거
                        clean_df = df_origin.dropna(subset=[target_col]).reset_index(drop=True)
                        X = clean_df[selected_features].copy()
                        y = clean_df[target_col].copy()

                        # 2. Y 인코딩 (문자열일 경우 숫자 변환)
                        le_target = None
                        if y.dtype == 'object':
                            le_target = LabelEncoder()
                            y = pd.Series(le_target.fit_transform(y), index=y.index)
                        
                        # 3. X 전처리
                        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        imputer = SimpleImputer(strategy='mean')
                        scaler = StandardScaler()
                        encoders = {}
                        
                        if num_cols:
                            X[num_cols] = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X[num_cols])), columns=num_cols, index=X.index)
                        
                        for col in cat_cols:
                            X[col] = X[col].fillna("Unknown").astype(str)
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col])
                            encoders[col] = le
                        
                        X = X.replace([np.inf, -np.inf], np.nan).fillna(0) # 잔여 결측 0 처리
                        
                        # 저장
                        st.session_state.preprocess.update({
                            "feature_cols": num_cols + cat_cols,
                            "imputer": imputer, "scaler": scaler, "encoders": encoders,
                            "target_col": target_col, "target_encoder": le_target
                        })
                        st.session_state.data["X_processed"] = X
                        st.session_state.data["y_processed"] = y
                        
                        st.success(f"✅ 전처리 완료! (데이터: {len(X)}행)")
                        st.dataframe(X.head())
                    except Exception as e:
                        st.error(f"오류: {e}")

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

# ==============================================================================
#  단계 4：모델 예측
# ==============================================================================
elif st.session_state.step == 4:
    st.subheader("🎯 모델 예측")
    if st.session_state.models["regression"] is None:
        st.warning("⚠️ 먼저 [모델 학습] 단계를 완료하세요")
    else:
        st.markdown("#### 예측 방식 선택")
        input_method = st.radio("", ["값 직접 입력", "CSV 파일 업로드"])
        
        # 예측 함수
        def run_prediction(input_df):
            # 전처리 적용
            prep = st.session_state.preprocess
            X = input_df.copy()
            
            # (간단화를 위해 누락된 컬럼 0 처리 및 순서 맞춤)
            for col in prep["feature_cols"]:
                if col not in X.columns: X[col] = 0
            X = X[prep["feature_cols"]] # 컬럼 순서 정렬
            
            # 모델 호출
            reg = st.session_state.models["regression"]
            dt = st.session_state.models["decision_tree"]
            w = st.session_state.models["mixed_weights"]
            
            if st.session_state.task == "logit":
                p1 = reg.predict_proba(X)[:, 1]
                p2 = dt.predict_proba(X)[:, 1]
                final_prob = (p1 * w["regression"]) + (p2 * w["decision_tree"])
                final_pred = (final_prob >= 0.5).astype(int)
                return final_pred, final_prob
            else:
                p1 = reg.predict(X)
                p2 = dt.predict(X)
                final_pred = (p1 * w["regression"]) + (p2 * w["decision_tree"])
                return final_pred, None

        if input_method == "값 직접 입력":
            st.info("주요 변수 5개만 입력 (나머지는 0 처리)")
            input_data = {}
            cols = st.columns(5)
            feats = st.session_state.preprocess["feature_cols"][:5]
            
            with st.form("input_form"):
                for i, f in enumerate(feats):
                    with cols[i]:
                        input_data[f] = st.number_input(f, value=0.0)
                if st.form_submit_button("예측"):
                    df_in = pd.DataFrame([input_data])
                    pred, prob = run_prediction(df_in)
                    st.metric("예측 결과", f"{pred[0]:.4f}")
        else:
            up = st.file_uploader("예측할 CSV 파일", type="csv")
            if up and st.button("일괄 예측"):
                df_batch = pd.read_csv(up)
                pred, prob = run_prediction(df_batch)
                df_batch["Prediction"] = pred
                st.dataframe(df_batch.head())

# ==============================================================================
#  단계 5：성능 평가
# ==============================================================================
elif st.session_state.step == 5:
    st.subheader("📈 모델 성능 평가")
    if st.session_state.models["regression"] is None:
        st.warning("⚠️ 먼저 [모델 학습] 단계를 완료하세요")
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        w = st.session_state.models["mixed_weights"]
        
        reg = st.session_state.models["regression"]
        dt = st.session_state.models["decision_tree"]
        
        # 현재 설정된 모델 구성 표시
        st.info(f"ℹ️ 현재 평가 모델 구성 - Logic: {w['regression']*100:.0f}% / Tree: {w['decision_tree']*100:.0f}%")

        if st.session_state.task == "logit":
            # 분류 평가
            p_reg = reg.predict_proba(X_test)[:, 1]
            p_dt = dt.predict_proba(X_test)[:, 1]
            p_mix = (p_reg * w["regression"]) + (p_dt * w["decision_tree"])
            pred_mix = (p_mix >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, pred_mix)
            try:
                roc_auc = auc(*roc_curve(y_test, p_mix)[:2])
            except:
                roc_auc = 0.0
            
            col1, col2 = st.columns(2)
            col1.metric("정확도 (Accuracy)", f"{acc:.2%}")
            col2.metric("AUC Score", f"{roc_auc:.3f}")
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, p_mix)
            fig = px.area(x=fpr, y=tpr, title="ROC Curve", labels=dict(x="FPR", y="TPR"))
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig, width='stretch')
            
        else:
            # 회귀 평가
            p_reg = reg.predict(X_test)
            p_dt = dt.predict(X_test)
            p_mix = (p_reg * w["regression"]) + (p_dt * w["decision_tree"])
            
            mae = mean_absolute_error(y_test, p_mix)
            r2 = r2_score(y_test, p_mix)
            
            col1, col2 = st.columns(2)
            col1.metric("MAE (오차)", f"{mae:.4f}")
            col2.metric("R² (설명력)", f"{r2:.4f}")
            
            fig = px.scatter(x=y_test, y=p_mix, title="Actual vs Predicted")
            fig.add_shape(type='line', line=dict(dash='dash', color='red'), x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max())
            st.plotly_chart(fig, width='stretch')
