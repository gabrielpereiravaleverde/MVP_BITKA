import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
import shap

# Define the app title
st.title("Análise Preditiva do Teor de Cobre")

# Generate some example data
np.random.seed(0)
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
X = np.random.rand(100, 5) * 9 + 1
y = 2 * X[:, 0] + 1.5 * X[:, 1] + 1 * X[:, 2] + 0.5 * X[:, 3] + 0.2 * X[:, 4] + np.random.randn(100)

# Create a DataFrame with the time series data
data = pd.DataFrame({'Date': dates, 'X1': X[:, 0], 'X2': X[:, 1], 'X3': X[:, 2], 'X4': X[:, 3], 'X5': X[:, 4], 'Y': y})
data.set_index('Date', inplace=True)

# Create a plot of the time series of Y using Plotly (at the top)
# st.header("Time Series of Y")

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, 
                         y=data['Y'], 
                         mode='lines+markers',line=dict(color='#247454'), name='Y'))
fig.update_xaxes(title_text='Data')
fig.update_yaxes(title_text='Teor de Cobre %')
fig.update_layout(title='Teor de Cobre')
st.plotly_chart(fig)

# Select a Date input
selected_date_input = st.date_input("Selecione uma data:", dates[0], min_value=dates[0], max_value=dates[-1])
selected_date = pd.to_datetime(selected_date_input)  # Convert selected date to the correct format

# Check if selected_date is in the dataset
if selected_date in data.index:
    selected_day_data = data.loc[selected_date]

    st.write(f"Teor no dia selecionado: {selected_day_data['Y']:.2f}%")

    # Title for the Inputs section
    st.header("Variáveis de Processo no dia selecionado")  # Variables on the selected day

    # Create columns to control the width of the input fields
    col1, col2, col3, col4 = st.columns(4)

    # X inputs (below the Teor de Cu no dia selecionado)
    with col1:
        x1_input = st.number_input("Reagente 1", 1.0, 10.0, value=float(selected_day_data['X1']))
        x5_input = st.number_input("Válvula 2", 1.0, 10.0, value=float(selected_day_data['X5']))
    with col2:
        x2_input = st.number_input("Reagente 2", 1.0, 10.0, value=float(selected_day_data['X2']))
    with col3:
        x3_input = st.number_input("Reagente 3", 1.0, 10.0, value=float(selected_day_data['X3']))
    with col4:
        x4_input = st.number_input("Válvula 1", 1.0, 10.0, value=float(selected_day_data['X4']))

    # Create and fit the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)

    # Predict button
    if st.button("Previsão do Teor"):
        # Predict Y based on the selected X values
        selected_x = np.array([x1_input, x2_input, x3_input, x4_input, x5_input]).reshape(1, -1)
        predicted_y = model.predict(selected_x)
        st.write(f"Teor Predito: {predicted_y[0]:.2f}%")

        # SHAP Waterfall Plot
        st.header("Impacto das variáveis da flotação no modelo")
        explainer = shap.Explainer(model)
        shap_values = explainer(selected_x)

        valores = np.hstack([shap_values.base_values, shap_values.values, shap_values.base_values])
        valores = valores[0]
        valores[6] = 0

        str_valores = [str(round(x, 1)) for x in valores]
        str_valores[-1] = round(sum(valores), 1)

        fig = go.Figure(go.Waterfall(
            name="20", orientation="v",
            measure=["relative", "relative", "relative", "relative", "relative", "relative", "total"],
            x=['Teor de Cu Médio', 'Reagente 1', 'Reagente 2', 'Reagente 3', 'Válvula 1', 'Válvula 2', 'Teor de Cu Predito'],
            textposition="outside",
            text=str_valores,
            y=valores,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        fig.update_layout(
            title="Nível de impacto das variáveis",
            showlegend=True,
            yaxis=dict(range=[0, max(y)])  # Set y-axis range
        )

        # Display the Plotly figure
        st.plotly_chart(fig)

else:
    st.warning("Selected date not found in the dataset.")