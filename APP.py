import streamlit as st
import pandas as pd
import joblib

# 设置页面标题
# st.title('An Explainable Prediction Model for Adverse Outcomes in Severe Scrub Typhus')

# 使用 st.columns 创建两列布局
left_column, right_column = st.columns(2)

# 在左侧列放置侧边栏的输入选项
with left_column:
    # st.header('Variables')
    a = st.number_input("Lymphocyte Count(10^9/L)", min_value=0.000, max_value=1000.000, step=0.001)
    b = st.number_input("Neutrophil To Lymphocyte Ratio", min_value=0.000, max_value=1000.000, step=0.001)
    c = st.number_input("Monocytes Counted(10^12/L)", min_value=0.000, max_value=1000.000, step=0.001)
    d = st.number_input("Platelet Count(10^9/L)", min_value=0.00, max_value=1000.00, step=0.01)
    e = st.number_input("Glucose(mmol/L)", min_value=0.000, max_value=1000.000, step=0.001)
    f = st.number_input("Systolic Slood Pressure(mmHg)", min_value=0.0, max_value=1000.0, step=0.1)
    g = st.number_input("National Institutes of Health Stroke Scale(0-42)", min_value=0, max_value=42, step=1)
    h = st.number_input("Modified Rankin Scale(0-6)", min_value=0, max_value=6, step=1)
# 如果按下按钮
if right_column.button("Predict"):  # 在右侧列放置预测按钮
    # 加载训练好的模型
    model = joblib.load("RF.pkl")

    # 将输入存储为 DataFrame，确保列名与模型训练时一致
    X = pd.DataFrame([[a, b, c, d, e, f, 1 if g == 0 else 2 if g <= 4 else 3 if g <= 15 else 4 if g <= 20 else 5,0 if h <=1 else 1]],
                     columns=['Lymphocyte Count(10^9/L)', 'Neutrophil To Lymphocyte Ratio',
                              'Monocytes Counted(10^12/L)',
                              'Platelet Count(10^9/L)', 'Glucose(mmol/L)', 'Systolic Slood Pressure(mmHg)',
                              'National Institutes of Health Stroke Scale(0-42)', 'Modified Rankin Scale(0-6)'])

    # 创建一个空白容器来放置预测结果
    result_container = st.empty()

    # 进行预测
    prediction = model.predict(X)[0]
    Predict_proba = model.predict_proba(X)[:, 1][0]

    # 输出预测结果到空白容器
    result_container.subheader(f"Probability of predicting adverse outcome: {'%.2f' % (Predict_proba * 100)}%")
