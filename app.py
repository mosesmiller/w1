import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set Streamlit page config
st.set_page_config(page_title="Linear Regression Confidence Visualizer", layout="centered")
st.title("📊 Linear Regression Simulation with Confidence Window")

# Sidebar inputs
st.sidebar.header("Simulation Parameters")
true_intercept = st.sidebar.number_input("True Intercept (Constant)", value=5.0)
true_slope = st.sidebar.number_input("True Slope", value=2.0)
n_samples = st.sidebar.slider("Number of Samples", min_value=10, max_value=1000, value=100, step=10)
noise_std = st.sidebar.slider("Noise Standard Deviation", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
bootstrap_draws = 30

# Generate original synthetic data
np.random.seed(42)
X = np.random.rand(n_samples, 1) * 10
noise = np.random.randn(n_samples, 1) * noise_std
y = true_intercept + true_slope * X + noise

# Fit base model
base_model = LinearRegression()
base_model.fit(X, y)
y_pred = base_model.predict(X)
estimated_intercept = base_model.intercept_[0]
estimated_slope = base_model.coef_[0][0]

# Show main regression output
st.subheader("📈 Single Regression Fit")
st.write(f"**True Intercept:** {true_intercept:.2f}")
st.write(f"**Estimated Intercept:** {estimated_intercept:.2f}")
st.write(f"**True Slope:** {true_slope:.2f}")
st.write(f"**Estimated Slope:** {estimated_slope:.2f}")

# Plot main regression
fig1, ax1 = plt.subplots()
ax1.scatter(X, y, label="Data", alpha=0.6)
ax1.plot(X, y_pred, color='red', label="Fitted Line")
ax1.set_xlabel("X")
ax1.set_ylabel("y")
ax1.set_title("Generated Data and Fitted Regression Line")
ax1.legend()
st.pyplot(fig1)

# New section: Confidence window with 30 bootstrapped regressions
st.subheader("🔁 Simulated Confidence Window")

fig2, ax2 = plt.subplots()

# Draw 30 random samples and plot regression lines
x_range = np.linspace(0, 10, 100).reshape(-1, 1)

for _ in range(bootstrap_draws):
    # Random sampling with replacement
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_sample = X[indices]
    y_sample = y[indices]

    model = LinearRegression()
    model.fit(X_sample, y_sample)
    y_bootstrap = model.predict(x_range)

    ax2.plot(x_range, y_bootstrap, color='gray', alpha=0.3)

# Overlay the true line
y_true_line = true_intercept + true_slope * x_range
ax2.plot(x_range, y_true_line, color='red', label="True Regression Line")

ax2.set_xlabel("X")
ax2.set_ylabel("y")
ax2.set_title(f"{bootstrap_draws} Bootstrap Regression Lines")
ax2.legend()
st.pyplot(fig2)
