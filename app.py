import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Fuzzy TOPSIS for Offshore Platform", layout="wide")

st.title("Поддержка принятия решений: Fuzzy TOPSIS для верхнего строения МНГС платформы")

# --- Определение критериев ---
criteria = ["ЧДД (млн руб.)", "ВНД (%)", "Масса ВСП (т)", "Срок строительства (мес.)", "Пиковая мощность (МВт)", "Выбросы CO2 (т)"]
criteria_type = ["benefit", "benefit", "cost", "cost", "benefit", "cost"]

# --- Лингвистические предпочтения ЛПР ---
st.subheader("Лингвистические предпочтения по важности критериев")
linguistic_scale = {
    "Очень низкая": (0, 0, 0.1),
    "Низкая": (0, 0.1, 0.3),
    "Средняя": (0.2, 0.5, 0.8),
    "Высокая": (0.7, 0.9, 1.0),
    "Очень высокая": (0.9, 1.0, 1.0)
}

weights_linguistic = []
cols = st.columns(len(criteria))
for i, crit in enumerate(criteria):
    with cols[i]:
        choice = st.selectbox(f"Важность: {crit}", list(linguistic_scale.keys()), key=crit)
        weights_linguistic.append(linguistic_scale[choice])

# --- Ввод данных альтернатив ---
st.subheader("Оценки альтернатив в виде треугольных нечетких чисел")
num_alts = st.number_input("Количество альтернатив", min_value=2, max_value=10, value=3)

alt_data = {}
for crit in criteria:
    alt_data[crit] = []
    for i in range(num_alts):
        col1, col2, col3 = st.columns(3)
        with col1:
            l = st.number_input(f"{crit} | A{i+1} (L)", key=f"{crit}_L_{i}")
        with col2:
            m = st.number_input(f"{crit} | A{i+1} (M)", key=f"{crit}_M_{i}")
        with col3:
            u = st.number_input(f"{crit} | A{i+1} (U)", key=f"{crit}_U_{i}")
        alt_data[crit].append((l, m, u))

# --- Функции для Fuzzy TOPSIS ---
def normalize_fuzzy_matrix(fuzzy_matrix, criteria_type):
    norm_matrix = []
    for j in range(len(criteria_type)):
        col = [row[j] for row in fuzzy_matrix]
        l_list = [x[0] for x in col]
        m_list = [x[1] for x in col]
        u_list = [x[2] for x in col]

        if criteria_type[j] == "benefit":
            max_u = max(u_list)
            norm_col = [(x[0]/max_u, x[1]/max_u, x[2]/max_u) for x in col]
        else:
            min_l = min(l_list)
            norm_col = [(min_l/x[2], min_l/x[1], min_l/x[0]) for x in col]

        norm_matrix.append(norm_col)
    # Транспонируем обратно
    return list(map(list, zip(*norm_matrix)))

def fuzzy_weighted_matrix(norm_matrix, weights):
    weighted_matrix = []
    for i in range(len(norm_matrix)):
        row = []
        for j in range(len(norm_matrix[0])):
            a, b, c = norm_matrix[i][j]
            w1, w2, w3 = weights[j]
            row.append((a*w1, b*w2, c*w3))
        weighted_matrix.append(row)
    return weighted_matrix

def fuzzy_ideal_solutions(weighted_matrix):
    tp = []
    tn = []
    for j in range(len(weighted_matrix[0])):
        col = [row[j] for row in weighted_matrix]
        l_list = [x[0] for x in col]
        m_list = [x[1] for x in col]
        u_list = [x[2] for x in col]
        tp.append((max(l_list), max(m_list), max(u_list)))
        tn.append((min(l_list), min(m_list), min(u_list)))
    return tp, tn

def distance(a, b):
    return np.sqrt((1/3) * ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2))

def closeness_coefficients(matrix, tp, tn):
    cc = []
    for row in matrix:
        d_pos = sum([distance(a, b) for a, b in zip(row, tp)])
        d_neg = sum([distance(a, b) for a, b in zip(row, tn)])
        cc.append(d_neg / (d_neg + d_pos))
    return cc

# --- Выполнение Fuzzy TOPSIS ---
if st.button("Выполнить расчет Fuzzy TOPSIS"):
    fuzzy_matrix = list(zip(*[alt_data[crit] for crit in criteria]))
    norm_matrix = normalize_fuzzy_matrix(fuzzy_matrix, criteria_type)
    weighted_matrix = fuzzy_weighted_matrix(norm_matrix, weights_linguistic)
    tp, tn = fuzzy_ideal_solutions(weighted_matrix)
    cc = closeness_coefficients(weighted_matrix, tp, tn)

    st.subheader("Итоговое ранжирование альтернатив")
    for i, score in enumerate(cc):
        st.write(f"Альтернатива A{i+1}: Коэффициент близости = {score:.4f}")

    best = np.argmax(cc) + 1
    st.success(f"Наилучшая альтернатива: A{best}")
