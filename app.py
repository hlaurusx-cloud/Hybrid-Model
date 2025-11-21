import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, auc, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from lightgbm import LGBMClassifier, LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

# ----------------------
# 1. 页面基础配置
# ----------------------
st.set_page_config(
    page_title="混合模型（하이브리드모형）动态框架",
    page_icon="📊",
    layout="wide"
)

# 全局状态管理（存储各步骤数据/模型，避免刷新丢失）
if "step" not in st.session_state:
    st.session_state.step = 0  # 0:初始页 1:数据上传 2:数据预处理 3:模型训练 4:预测 5:评估
if "data" not in st.session_state:
    st.session_state.data = {"accept": None, "genied": None, "merged": None}
if "preprocess" not in st.session_state:
    st.session_state.preprocess = {"imputer": None, "scaler": None, "encoders": None, "feature_cols": None, "target_col": None}
if "models" not in st.session_state:
    st.session_state.models = {"lr": None, "lgb": None, "mixed_weights": {"lr": 0.3, "lgb": 0.7}}
if "task" not in st.session_state:
    st.session_state.task = "分类"  # 默认为分类，可切换为回归

# ----------------------
# 2. 侧边栏：步骤导航 + 核心配置
# ----------------------
st.sidebar.title("📌 混合模型操作流程")
st.sidebar.divider()

# 步骤导航按钮
steps = ["初始设置", "上传数据", "数据预处理", "模型训练", "模型预测", "效果评估"]
for i, step_name in enumerate(steps):
    if st.sidebar.button(step_name, key=f"btn_{i}"):
        st.session_state.step = i

# 核心配置（任务类型 + 混合权重）
st.sidebar.divider()
st.sidebar.subheader("核心配置")
st.session_state.task = st.sidebar.radio("任务类型", options=["分类", "回归"], index=0)

if st.session_state.step >= 3:  # 模型训练后可调整权重
    st.sidebar.subheader("混合模型权重")
    lr_weight = st.sidebar.slider(
        "逻辑回归权重（可解释性）",
        min_value=0.0, max_value=1.0, value=st.session_state.models["mixed_weights"]["lr"], step=0.1
    )
    st.session_state.models["mixed_weights"]["lr"] = lr_weight
    st.session_state.models["mixed_weights"]["lgb"] = 1 - lr_weight
    st.sidebar.text(f"LightGBM权重（高精度）：{1 - lr_weight:.1f}")

# ----------------------
# 3. 主页面：分步骤展示内容
# ----------------------
st.title("📊 混合模型（하이브리드모형）动态部署框架")
st.markdown("**支持上传 accept/genied 原始数据，一键完成预处理→训练→预测全流程**")
st.divider()

# ----------------------
# 步骤0：初始设置（引导页）
# ----------------------
if st.session_state.step == 0:
    st.subheader("🎉 欢迎使用混合模型动态框架")
    st.markdown("""
    本框架支持 **收到数据后直接导入使用**，无需提前预处理或训练模型，核心流程如下：
    
    1. **上传数据**：上传 accept 和 genied 两个原始文件（CSV/Parquet/Excel）
    2. **数据预处理**：合并数据、填充缺失值、编码类别特征
    3. **模型训练**：训练「逻辑回归+LightGBM」混合模型
    4. **模型预测**：支持单条输入或批量上传预测
    5. **效果评估**：对比混合模型与单一模型的性能
    
    ### 适用场景
    - 分类任务（如：预测用户是否接受服务、是否违约）
    - 回归任务（如：预测销量、金额、评分）
    
    ### 点击左侧「上传数据」开始使用！
    """)

# ----------------------
# 步骤1：上传数据（核心：支持动态导入两个原始文件）
# ----------------------
elif st.session_state.step == 1:
    st.subheader("📤 上传数据（accept + genied）")
    st.markdown("支持格式：CSV、Parquet、Excel（.xlsx/.xls）")
    
    col1, col2 = st.columns(2)
    
    # 上传 accept 文件
    with col1:
        st.markdown("### accept 数据集")
        accept_file = st.file_uploader("选择 accept 文件", type=["csv", "parquet", "xlsx", "xls"], key="accept")
        if accept_file is not None:
            # 读取不同格式文件
            if accept_file.name.endswith(".csv"):
                df_accept = pd.read_csv(accept_file)
            elif accept_file.name.endswith(".parquet"):
                df_accept = pd.read_parquet(accept_file)
            elif accept_file.name.endswith((".xlsx", ".xls")):
                df_accept = pd.read_excel(accept_file)
            st.session_state.data["accept"] = df_accept
            st.metric("数据量", f"{len(df_accept):,} 行 × {len(df_accept.columns)} 列")
            st.dataframe(df_accept.head(3), use_container_width=True)
    
    # 上传 genied 文件
    with col2:
        st.markdown("### genied 数据集")
        genied_file = st.file_uploader("选择 genied 文件", type=["csv", "parquet", "xlsx", "xls"], key="genied")
        if genied_file is not None:
            if genied_file.name.endswith(".csv"):
                df_genied = pd.read_csv(genied_file)
            elif genied_file.name.endswith(".parquet"):
                df_genied = pd.read_parquet(genied_file)
            elif genied_file.name.endswith((".xlsx", ".xls")):
                df_genied = pd.read_excel(genied_file)
            st.session_state.data["genied"] = df_genied
            st.metric("数据量", f"{len(df_genied):,} 行 × {len(df_genied.columns)} 列")
            st.dataframe(df_genied.head(3), use_container_width=True)
    
    # 数据合并（需用户指定关联键）
    st.divider()
    if st.session_state.data["accept"] is not None and st.session_state.data["genied"] is not None:
        st.markdown("### 数据合并设置")
        # 自动识别共同列作为关联键候选
        common_cols = list(set(st.session_state.data["accept"].columns) & set(st.session_state.data["genied"].columns))
        if common_cols:
            join_key = st.selectbox("选择关联键（用于合并两个数据集）", options=common_cols, index=0)
        else:
            join_key = st.text_input("无共同列，请输入关联键（需两个文件中均存在）")
        
        join_type = st.selectbox("合并方式", options=["内连接（只保留共同数据）", "左连接（保留accept全部数据）"], index=0)
        join_type_map = {"内连接（只保留共同数据）": "inner", "左连接（保留accept全部数据）": "left"}
        
        if st.button("开始合并数据"):
            try:
                df_merged = pd.merge(
                    st.session_state.data["accept"],
                    st.session_state.data["genied"],
                    on=join_key,
                    how=join_type_map[join_type]
                )
                st.session_state.data["merged"] = df_merged
                st.success(f"数据合并成功！合并后数据：{len(df_merged):,} 行 × {len(df_merged.columns)} 列")
                st.dataframe(df_merged.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"合并失败：{str(e)}")
    else:
        st.warning("请先上传两个数据集再进行合并")

# ----------------------
# 步骤2：数据预处理（动态适配数据，无需提前配置）
# ----------------------
elif st.session_state.step == 2:
    st.subheader("🧹 数据预处理")
    
    if st.session_state.data["merged"] is None:
        st.warning("请先完成「上传数据」步骤并合并数据")
    else:
        df_merged = st.session_state.data["merged"]
        
        # 1. 数据概览（缺失值、数据类型）
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 数据基本信息")
            st.write(f"总数据量：{len(df_merged):,} 行 × {len(df_merged.columns)} 列")
            st.write("数据类型分布：")
            st.dataframe(df_merged.dtypes.value_counts().reset_index(), use_container_width=True)
        
        with col2:
            st.markdown("### 缺失值分布")
            missing_info = df_merged.isnull().sum()[df_merged.isnull().sum() > 0].reset_index()
            missing_info.columns = ["字段名", "缺失值数量"]
            if len(missing_info) > 0:
                st.dataframe(missing_info, use_container_width=True)
                fig_missing = px.imshow(df_merged.isnull(), color_continuous_scale="Reds", title="缺失值热力图")
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("无缺失值！")
        
        # 2. 预处理配置（用户可调整）
        st.divider()
        st.markdown("### 预处理参数设置")
        
        # 选择目标列（预测变量）
        target_col = st.selectbox("选择目标列（需预测的变量）", options=df_merged.columns, index=-1)
        st.session_state.preprocess["target_col"] = target_col
        
        # 选择特征列（排除目标列和无用列）
        exclude_cols = st.multiselect("选择需排除的列（如ID、无关字段）", options=[col for col in df_merged.columns if col != target_col])
        feature_cols = [col for col in df_merged.columns if col not in exclude_cols + [target_col]]
        st.session_state.preprocess["feature_cols"] = feature_cols
        
        # 缺失值处理
        st.markdown("#### 缺失值处理")
        impute_strategy = st.selectbox("数值型缺失值填充方式", options=["中位数", "均值", "众数"], index=0)
        impute_strategy_map = {"中位数": "median", "均值": "mean", "众数": "most_frequent"}
        
        # 类别特征编码
        st.markdown("#### 类别特征编码")
        cat_encoding = st.selectbox("类别型特征编码方式", options=["标签编码（LabelEncoder）", "独热编码（OneHotEncoder）"], index=0)
        
        # 3. 执行预处理
        if st.button("开始预处理"):
            try:
                X = df_merged[feature_cols].copy()
                y = df_merged[target_col].copy()
                
                # 分离数值型和类别型特征
                num_cols = X.select_dtypes(include=["int64", "float64"]).columns
                cat_cols = X.select_dtypes(include=["object", "category"]).columns
                
                # 数值型预处理：缺失值填充 + 标准化
                imputer = SimpleImputer(strategy=impute_strategy_map[impute_strategy])
                X[num_cols] = imputer.fit_transform(X[num_cols])
                
                scaler = StandardScaler()
                X[num_cols] = scaler.fit_transform(X[num_cols])
                
                # 类别型预处理：缺失值填充 + 编码
                encoders = {}
                for col in cat_cols:
                    # 填充类别型缺失值为"未知"
                    X[col] = X[col].fillna("未知").astype(str)
                    
                    if cat_encoding == "标签编码（LabelEncoder）":
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                        encoders[col] = le
                    else:  # 独热编码
                        ohe = OneHotEncoder(sparse_output=False, drop="first")
                        ohe_result = ohe.fit_transform(X[[col]])
                        ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0][1:]]  # 排除第一个类别（避免共线性）
                        X = pd.concat([X.drop(col, axis=1), pd.DataFrame(ohe_result, columns=ohe_cols)], axis=1)
                        encoders[col] = (ohe, ohe_cols)
                
                # 保存预处理组件
                st.session_state.preprocess["imputer"] = imputer
                st.session_state.preprocess["scaler"] = scaler
                st.session_state.preprocess["encoders"] = encoders
                st.session_state.preprocess["feature_cols"] = list(X.columns)  # 更新后的特征列（含独热编码列）
                
                # 保存预处理后的数据
                st.session_state.data["X_processed"] = X
                st.session_state.data["y_processed"] = y
                
                st.success("数据预处理完成！")
                st.markdown(f"预处理后特征数：{len(X.columns)}")
                st.dataframe(X.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"预处理失败：{str(e)}")

# ----------------------
# 步骤3：模型训练（混合模型：逻辑回归+LightGBM）
# ----------------------
elif st.session_state.step == 3:
    st.subheader("🚀 混合模型训练")
    
    # 检查预处理是否完成
    if "X_processed" not in st.session_state.data or "y_processed" not in st.session_state.data:
        st.warning("请先完成「数据预处理」步骤")
    else:
        X = st.session_state.data["X_processed"]
        y = st.session_state.data["y_processed"]
        
        # 数据拆分（训练集+测试集）
        st.markdown("### 训练配置")
        test_size = st.slider("测试集占比", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if st.session_state.task == "分类" else None)
        
        # 模型选择（根据任务类型）
        if st.session_state.task == "分类":
            lr_model = LogisticRegression(max_iter=1000)
            lgb_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        else:  # 回归
            lr_model = LinearRegression()
            lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        
        # 训练模型
        if st.button("开始训练模型"):
            with st.spinner("模型训练中..."):
                # 训练单一模型
                lr_model.fit(X_train, y_train)
                lgb_model.fit(X_train, y_train)
                
                # 保存模型
                st.session_state.models["lr"] = lr_model
                st.session_state.models["lgb"] = lgb_model
                
                # 训练集/测试集预测
                st.session_state.data["X_train"] = X_train
                st.session_state.data["X_test"] = X_test
                st.session_state.data["y_train"] = y_train
                st.session_state.data["y_test"] = y_test
                
                st.success("模型训练完成！")
                st.markdown("✅ 已训练模型：")
                st.markdown("- 逻辑回归（可解释性强）")
                st.markdown("- LightGBM（高精度）")
                st.markdown("- 混合模型（加权融合前两者）")

# ----------------------
# 步骤4：模型预测（单条/批量上传）
# ----------------------
elif st.session_state.step == 4:
    st.subheader("🎯 模型预测")
    
    # 检查模型是否训练完成
    if st.session_state.models["lr"] is None or st.session_state.models["lgb"] is None:
        st.warning("请先完成「模型训练」步骤")
    else:
        # 预测函数（复用预处理逻辑）
        def predict(input_data):
            X = input_data.copy()
            preprocess = st.session_state.preprocess
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            
            # 数值型预处理
            X[num_cols] = preprocess["imputer"].transform(X[num_cols])
            X[num_cols] = preprocess["scaler"].transform(X[num_cols])
            
            # 类别型预处理
            for col in cat_cols:
                X[col] = X[col].fillna("未知").astype(str)
                encoder = preprocess["encoders"][col]
                
                if isinstance(encoder, LabelEncoder):
                    # 处理未见过的类别
                    X[col] = X[col].replace([x for x in X[col].unique() if x not in encoder.classes_], "未知")
                    if "未知" not in encoder.classes_:
                        encoder.classes_ = np.append(encoder.classes_, "未知")
                    X[col] = encoder.transform(X[col])
                else:  # OneHotEncoder
                    ohe, ohe_cols = encoder
                    ohe_result = ohe.transform(X[[col]])
                    X = pd.concat([X.drop(col, axis=1), pd.DataFrame(ohe_result, columns=ohe_cols)], axis=1)
            
            # 确保特征列顺序一致
            X = X[preprocess["feature_cols"]]
            
            # 混合模型预测
            lr_weight = st.session_state.models["mixed_weights"]["lr"]
            lgb_weight = st.session_state.models["mixed_weights"]["lgb"]
            
            if st.session_state.task == "分类":
                lr_proba = st.session_state.models["lr"].predict_proba(X)[:, 1]
                lgb_proba = st.session_state.models["lgb"].predict_proba(X)[:, 1]
                mixed_proba = lr_weight * lr_proba + lgb_weight * lgb_proba
                pred = (mixed_proba >= 0.5).astype(int)
                return pred, mixed_proba
            else:
                lr_pred = st.session_state.models["lr"].predict(X)
                lgb_pred = st.session_state.models["lgb"].predict(X)
                mixed_pred = lr_weight * lr_pred + lgb_weight * lgb_pred
                return mixed_pred, None
        
        # 预测方式选择
        predict_mode = st.radio("预测方式", options=["单条数据输入", "批量上传CSV"])
        
        # 单条输入预测
        if predict_mode == "单条数据输入":
            st.markdown("#### 单条数据输入（请填写特征值）")
            feature_cols = st.session_state.preprocess["feature_cols"]
            input_data = {}
            
            # 动态生成输入表单（根据特征类型）
            with st.form("single_pred_form"):
                cols = st.columns(3)
                for i, col in enumerate(feature_cols[:9]):  # 最多显示9个特征（避免界面拥挤）
                    with cols[i % 3]:
                        # 判断特征类型（数值/类别，基于预处理前的信息）
                        if col in st.session_state.data["X_processed"].select_dtypes(include=["int64", "float64"]).columns:
                            input_data[col] = st.number_input(col, value=0.0)
                        else:
                            # 类别特征：用训练集中的唯一值作为选项
                            unique_vals = st.session_state.data["X_processed"][col].unique()[:10]  # 最多10个选项
                            input_data[col] = st.selectbox(col, options=unique_vals)
                
                # 提交预测
                submit_btn = st.form_submit_button("开始预测")
            
            if submit_btn:
                input_df = pd.DataFrame([input_data])
                pred, proba = predict(input_df)
                
                st.divider()
                st.markdown("### 预测结果")
                if st.session_state.task == "分类":
                    st.metric("预测结果", "正类" if pred[0] == 1 else "负类")
                    st.metric("正类概率", f"{proba[0]:.3f}" if proba is not None else "-")
                else:
                    st.metric("预测结果", f"{pred[0]:.2f}")
        
        # 批量上传预测
        else:
            st.markdown("#### 批量上传CSV预测")
            uploaded_file = st.file_uploader("上传包含特征列的CSV文件", type=["csv"])
            
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                st.metric("上传数据量", f"{len(batch_df):,} 行")
                st.dataframe(batch_df.head(3), use_container_width=True)
                
                if st.button("开始批量预测"):
                    with st.spinner("预测中..."):
                        pred, proba = predict(batch_df)
                        batch_df["混合模型预测结果"] = pred
                        if proba is not None:
                            batch_df["正类概率"] = proba.round(3)
                        
                        st.divider()
                        st.markdown("### 批量预测结果")
                        st.dataframe(batch_df[["混合模型预测结果"] + (["正类概率"] if proba is not None else []) + feature_cols[:3]], use_container_width=True)
                        
                        # 下载结果
                        csv = batch_df.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button(
                            label="下载预测结果",
                            data=csv,
                            file_name="混合模型批量预测结果.csv",
                            mime="text/csv"
                        )

# ----------------------
# 步骤5：效果评估（混合模型 vs 单一模型）
# ----------------------
elif st.session_state.step == 5:
    st.subheader("📈 模型效果评估")
    
    if st.session_state.models["lr"] is None or st.session_state.models["lgb"] is None:
        st.warning("请先完成「模型训练」步骤")
    else:
        X_test = st.session_state.data["X_test"]
        y_test = st.session_state.data["y_test"]
        lr_model = st.session_state.models["lr"]
        lgb_model = st.session_state.models["lgb"]
        lr_weight = st.session_state.models["mixed_weights"]["lr"]
        lgb_weight = st.session_state.models["mixed_weights"]["lgb"]
        
        # 计算各模型预测结果
        if st.session_state.task == "分类":
            lr_pred = lr_model.predict(X_test)
            lgb_pred = lgb_model.predict(X_test)
            lr_proba = lr_model.predict_proba(X_test)[:, 1]
            lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
            mixed_proba = lr_weight * lr_proba + lgb_weight * lgb_proba
            mixed_pred = (mixed_proba >= 0.5).astype(int)
            
            # 计算分类指标
            def calc_class_metrics(y_true, y_pred, y_proba):
                acc = accuracy_score(y_true, y_pred)
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_score = auc(fpr, tpr)
                return {"准确率": acc, "AUC": auc_score}
            
            lr_metrics = calc_class_metrics(y_test, lr_pred, lr_proba)
            lgb_metrics = calc_class_metrics(y_test, lgb_pred, lgb_proba)
            mixed_metrics = calc_class_metrics(y_test, mixed_pred, mixed_proba)
            
            metrics_df = pd.DataFrame({
                "模型": ["逻辑回归", "LightGBM", "混合模型"],
                "准确率": [lr_metrics["准确率"], lgb_metrics["准确率"], mixed_metrics["准确率"]],
                "AUC": [lr_metrics["AUC"], lgb_metrics["AUC"], mixed_metrics["AUC"]]
            }).round(3)
        
        else:  # 回归
            lr_pred = lr_model.predict(X_test)
            lgb_pred = lgb_model.predict(X_test)
            mixed_pred = lr_weight * lr_pred + lgb_weight * lgb_pred
            
            # 计算回归指标
            def calc_reg_metrics(y_true, y_pred):
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                return {"MAE": mae, "RMSE": rmse, "R²": r2}
            
            lr_metrics = calc_reg_metrics(y_test, lr_pred)
            lgb_metrics = calc_reg_metrics(y_test, lgb_pred)
            mixed_metrics = calc_reg_metrics(y_test, mixed_pred)
            
            metrics_df = pd.DataFrame({
                "模型": ["逻辑回归", "LightGBM", "混合模型"],
                "MAE": [lr_metrics["MAE"], lgb_metrics["MAE"], mixed_metrics["MAE"]],
                "RMSE": [lr_metrics["RMSE"], lgb_metrics["RMSE"], mixed_metrics["RMSE"]],
                "R²": [lr_metrics["R²"], lgb_metrics["R²"], mixed_metrics["R²"]]
            }).round(3)
        
        # 展示指标对比
        st.markdown("### 模型性能对比")
        st.dataframe(metrics_df, use_container_width=True)
        
        # 可视化对比
        col1, col2 = st.columns(2)
        
        # 分类任务可视化
        if st.session_state.task == "分类":
            with col1:
                st.markdown("### ROC-AUC 曲线")
                fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
                fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_proba)
                fpr_mixed, tpr_mixed, _ = roc_curve(y_test, mixed_proba)
                
                fig_auc = go.Figure()
                fig_auc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, name=f"逻辑回归 (AUC={lr_metrics['AUC']:.3f})"))
                fig_auc.add_trace(go.Scatter(x=fpr_lgb, y=tpr_lgb, name=f"LightGBM (AUC={lgb_metrics['AUC']:.3f})"))
                fig_auc.add_trace(go.Scatter(x=fpr_mixed, y=tpr_mixed, name=f"混合模型 (AUC={mixed_metrics['AUC']:.3f})", line_dash="dash", line_width=3))
                fig_auc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="随机猜测", line_color="gray", line_dash="dot"))
                st.plotly_chart(fig_auc, use_container_width=True)
            
            with col2:
                st.markdown("### 混淆矩阵（混合模型）")
                cm = confusion_matrix(y_test, mixed_pred)
                cm_df = pd.DataFrame(cm, index=["真实负类", "真实正类"], columns=["预测负类", "预测正类"])
                fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
                st.plotly_chart(fig_cm, use_container_width=True)
        
        # 回归任务可视化
        else:
            with col1:
                st.markdown("### 预测值 vs 真实值（混合模型）")
                fig_pred = px.scatter(x=y_test, y=mixed_pred, title="真实值 vs 预测值", labels={"x": "真实值", "y": "预测值"})
                fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], line_color="red", name="理想拟合线"))
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                st.markdown("### 残差图（混合模型）")
                residuals = y_test - mixed_pred
                fig_res = px.scatter(x=mixed_pred, y=residuals, title="预测值 vs 残差", labels={"x": "预测值", "y": "残差"})
                fig_res.add_trace(go.Scatter(x=[mixed_pred.min(), mixed_pred.max()], y=[0, 0], line_color="red", name="残差=0线"))
                st.plotly_chart(fig_res, use_container_width=True)
        
        # 模型解释（特征重要性）
        st.divider()
        st.markdown("### 模型解释：核心特征重要性")
        feature_importance = pd.DataFrame({
            "特征名": st.session_state.preprocess["feature_cols"],
            "重要性": lgb_model.feature_importances_
        }).sort_values("重要性", ascending=False).head(10)
        
        fig_importance = px.bar(feature_importance, x="重要性", y="特征名", orientation="h", color="重要性", color_continuous_scale="viridis")
        st.plotly_chart(fig_importance, use_container_width=True)
