import streamlit as st
import pandas as pd
import plotly.express as px

# Example data for each metric
metrics_data = {
    'Time': [10, 15, 20, 25, 30],
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'Relevance': [0.8, 0.9, 0.7, 0.85, 0.75],
    'Groundedness': [0.7, 0.85, 0.9, 0.8, 0.75],
    'Sentiment': [0.6, 0.8, 0.5, 0.7, 0.9],
    'Model Agreement': [0.75, 0.8, 0.85, 0.9, 0.7],
    'Language Match': [0.8, 0.7, 0.9, 0.75, 0.85],
    'Toxicity': [0.1, 0.2, 0.15, 0.25, 0.3],
    'Moderation': [0.9, 0.85, 0.7, 0.8, 0.75],
    'Stereotypes': [0.2, 0.15, 0.3, 0.25, 0.1],
    'Summarization': [0.85, 0.8, 0.9, 0.7, 0.75],
    'Embeddings Distance': [0.5, 0.6, 0.4, 0.7, 0.8],
    'Entity Recognition': [0.8, 0.75, 0.9, 0.85, 0.7],
    'Coherence': [0.9, 0.85, 0.75, 0.8, 0.7],
    'Completeness': [0.85, 0.8, 0.9, 0.7, 0.75],
    'Clarity': [0.9, 0.85, 0.7, 0.8, 0.75],
    'Novelty': [0.8, 0.75, 0.9, 0.85, 0.7],
    'Intent Understanding': [0.85, 0.9, 0.8, 0.7, 0.75],
    'Ambiguity Handling': [0.9, 0.8, 0.75, 0.85, 0.7],
    'User Engagement': [0.7, 0.8, 0.9, 0.85, 0.75],
    'Error Rate': [0.1, 0.2, 0.15, 0.25, 0.3],
    'Adaptability': [0.9, 0.85, 0.7, 0.8, 0.75],
    'Bias Detection': [0.8, 0.75, 0.9, 0.85, 0.7],
    'Ambient Context Awareness': [0.85, 0.9, 0.7, 0.8, 0.75],
    'Ethical Considerations': [0.7, 0.8, 0.9, 0.85, 0.75],
}

# Create a DataFrame from the example data
df = pd.DataFrame(metrics_data)

# Streamlit App
st.title("Language Model Metrics Dashboard")

# Data Summary
st.subheader("Data Summary")
st.write("This dashboard visualizes various metrics related to the performance of a language model over time.")

# Sidebar with metric selection and user input forms
selected_metric = st.sidebar.selectbox("Select Metric", df.columns)

start_date = st.sidebar.date_input("Start Date", min(df['Date']), max(df['Date']))
end_date = st.sidebar.date_input("End Date", min(df['Date']), max(df['Date']))
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

chart_type = st.sidebar.selectbox("Select Chart Type", ["line", "bar", "scatter"])

# Display the selected metric chart
st.subheader(f"Chart for {selected_metric}")
fig = px.scatter(filtered_df, x='Date', y=selected_metric, title=f"{selected_metric} Over Time")
fig.update_traces(hovertemplate="Date: %{x}<br>Value: %{y}")

# Visualization Customization
st.sidebar.subheader("Chart Customization")
color_option = st.sidebar.checkbox("Use Custom Colors")
if color_option:
    color = st.sidebar.color_picker("Select Color", value='blue')
    fig.update_traces(marker=dict(color=color))

# Display the chart with or without customization
st.plotly_chart(fig, use_container_width=True)

# Interactive Elements
st.subheader("Interactive Elements")
st.write("Hover over the chart to see details. You can customize the chart on the sidebar.")

# Error Handling
if selected_metric not in df.columns:
    st.error("Selected metric not found in the dataset. Please choose a valid metric.")

