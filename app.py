import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(page_title="Linear Regression Simulator", layout="centered")

st.title("🔢 Linear Regression Simulation")

# User inputs
st.sidebar.header("Simulation Parameters")
true_intercept = st.sidebar.number_input("True Intercept (Constant)", value=5.0)
true_slope = st.sidebar.number_input("True Slope", value=2.0)
n_samples = st.sidebar.slider("Number of Samples", min_value=10, max_value=1000, value=100, step=10)
noise_std = st.sidebar.slider("Noise Standard Deviation", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(n_samples, 1) * 10  # Features between 0 and 10
noise = np.random.randn(n_samples, 1) * noise_std
y = true_intercept + true_slope * X + noise

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
estimated_intercept = model.intercept_[0]
estimated_slope = model.coef_[0][0]
y_pred = model.predict(X)

# Display results
st.subheader("📈 Regression Results")
st.write(f"**True Intercept:** {true_intercept:.2f}")
st.write(f"**Estimated Intercept:** {estimated_intercept:.2f}")
st.write(f"**True Slope:** {true_slope:.2f}")
st.write(f"**Estimated Slope:** {estimated_slope:.2f}")

# Plot
fig, ax = plt.subplots()
ax.scatter(X, y, label="Data", alpha=0.6)
ax.plot(X, y_pred, color='red', label="Fitted Line")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title("Generated Data and Fitted Regression Line")
ax.legend()
st.pyplot(fig)
