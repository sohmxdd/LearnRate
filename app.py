import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="LearnRate", page_icon="⚡", layout="centered")

# Styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            background: linear-gradient(90deg, #00dbde, #fc00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }
        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(8px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

#Session State
if "alpha" not in st.session_state:
    st.session_state["alpha"] = 0.01
if "iterations" not in st.session_state:
    st.session_state["iterations"] = 1000
if "noise" not in st.session_state:
    st.session_state["noise"] = 0.5

st.markdown(
    """
    <h1 style='font-size: 2.8rem;'>
        ⚡ LearnRate: Linear Regression from Scratch
    </h1>
    """, 
    unsafe_allow_html=True
)

st.write("""
Welcome to **LearnRate**, an interactive app that demonstrates 
**Linear Regression** implemented from scratch using **Gradient Descent**.  
Use the sidebar to adjust the learning rate, iterations, and dataset noise.
""")

#Sidebar Controls
st.sidebar.header("🔧 Controls")
alpha = st.sidebar.slider("Learning Rate (α)", 0.001, 0.1, st.session_state["alpha"], 0.001)
iterations = st.sidebar.slider("Iterations", 100, 5000, st.session_state["iterations"], 100)

add_noise = st.sidebar.checkbox("Add Noise to Data", value=True)
noise = st.sidebar.slider("Dataset Noise", 0.0, 3.0, st.session_state["noise"], 0.1) if add_noise else 0.0

#Generate Data
np.random.seed(42)
true_w = 3
true_b = 4
x = np.linspace(0, 10, 50)
y = true_w * x + true_b + np.random.randn(50) * noise

#Functions
def compute_cost(x, y, w, b):
    m = len(x)
    return sum((w * x[i] + b - y[i]) ** 2 for i in range(m)) / (2 * m)

def compute_gradient(x, y, w, b):
    m = len(x)
    dj_dw = sum((w * x[i] + b - y[i]) * x[i] for i in range(m)) / m
    dj_db = sum((w * x[i] + b - y[i]) for i in range(m)) / m
    return dj_dw, dj_db

def gradient_descent_animated(x, y, w_in, b_in, alpha, num_iter):
    w = w_in
    b = b_in
    w_history, b_history, cost_history = [], [], []
    for i in range(num_iter):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        if i % 10 == 0:
            cost = compute_cost(x, y, w, b)
            cost_history.append(cost)
            w_history.append(w)
            b_history.append(b)
    return w, b, cost_history, w_history, b_history

# Run Gradient Descent
initial_w, initial_b = 0, 0
w_learned, b_learned, J_history, w_hist, b_hist = gradient_descent_animated(
    x, y, initial_w, initial_b, alpha, iterations
)

#Graph 1: Matplotlib
fig, ax = plt.subplots()
ax.scatter(x, y, color='blue', label='Training Data')
ax.plot(x, true_w * x + true_b, color='red', linestyle='--', label='True Line')
ax.set_title("Synthetic Data")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
st.pyplot(fig)

#raph 2: Plotly Animation
frames = [
    go.Frame(data=[go.Scatter(x=x, y=w_hist[i] * x + b_hist[i],
                              mode='lines', line=dict(color='green', width=3))],
             name=f"frame{i}")
    for i in range(len(w_hist))
]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x, y=y, mode='markers',
    marker=dict(color='blue', size=8, line=dict(width=1, color='white')),
    name='Training Data'))
fig2.add_trace(go.Scatter(x=x, y=true_w * x + true_b,
    mode='lines', line=dict(color='red', dash='dash'), name='True Line'))
fig2.add_trace(go.Scatter(x=x, y=w_hist[0] * x + b_hist[0],
    mode='lines', line=dict(color='green', width=3), name='Learned Line'))
fig2.update(frames=frames)
fig2.update_layout(
    title='📊 Learned Regression Fit (Animated)',
    xaxis_title='x',
    yaxis_title='y',
    updatemenus=[{
        "buttons": [
            {"args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
             "label": "▶ Play", "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
             "label": "⏸ Pause", "method": "animate"}
        ],
        "direction": "right",
        "type": "buttons",
        "x": 1.0, "xanchor": "left", "y": 1.15, "yanchor": "top",
        "showactive": False, "font": dict(color="white", size=12),
        "bgcolor": "rgba(255,255,255,0.1)",
        "borderwidth": 1, "bordercolor": "white"
    }],
    template='plotly_dark'
)
st.plotly_chart(fig2, use_container_width=True)

#Graph 3: Plotly Cost
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=list(range(0, iterations, 10)), y=J_history,
    mode='lines+markers', line=dict(color='purple', width=3),
    marker=dict(size=6, color='white', line=dict(width=2, color='purple')),
    name='Cost'))
fig3.update_layout(
    title='📉 Cost vs Iterations',
    xaxis_title='Iterations',
    yaxis_title='Cost',
    template='plotly_dark'
)
st.plotly_chart(fig3, use_container_width=True)

# Learned Parameters
st.success(f"Learned Weight (w): {w_learned:.4f}")
st.success(f"Learned Bias (b): {b_learned:.4f}")
st.info(f"Final Cost: {J_history[-1]:.4f}")
