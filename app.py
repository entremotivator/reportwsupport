import streamlit as st
import pandas as pd
import plotly
import plotly.express as px

# Example data for language model metrics
language_metrics_data = {
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

# Create a DataFrame for language model metrics
df_language = pd.DataFrame(language_metrics_data)

# Example data for image model metrics
image_metrics_data = {
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'Accuracy': [0.85, 0.88, 0.82, 0.90, 0.87],
    'Precision': [0.78, 0.85, 0.80, 0.88, 0.82],
    'Recall': [0.90, 0.92, 0.85, 0.94, 0.91],
    'F1 Score': [0.83, 0.88, 0.82, 0.91, 0.86],
    'IoU': [0.75, 0.82, 0.78, 0.85, 0.80],
    'Top-1 Accuracy': [0.75, 0.82, 0.78, 0.85, 0.80],
    'Top-5 Accuracy': [0.92, 0.94, 0.90, 0.95, 0.93],
    'mAP': [0.80, 0.85, 0.78, 0.88, 0.82],
    'Structural Similarity Index (SSI)': [0.88, 0.90, 0.85, 0.92, 0.89],
    'Peak Signal-to-Noise Ratio (PSNR)': [22.5, 23.1, 21.8, 24.0, 22.7],
    'Robustness': [0.95, 0.92, 0.97, 0.91, 0.94],
    'Latency (ms)': [120, 110, 125, 105, 115],
}

# Create a DataFrame for image model metrics
df_image_metrics = pd.DataFrame(image_metrics_data)

# Streamlit App
st.title("Model Metrics Dashboard")

# Language Model Metrics Section
st.header("Language Model Metrics")
st.subheader("Data Summary")
st.write("This section visualizes various metrics related to the performance of a language model over time.")

# Sidebar for language model metric selection
selected_language_metric = st.sidebar.selectbox("Select Language Metric", df_language.columns)

# Date range selection for language model metrics
start_date_lang = st.sidebar.date_input("Start Date", min(df_language['Date']), max(df_language['Date']))
end_date_lang = st.sidebar.date_input("End Date", min(df_language['Date']), max(df_language['Date']))
filtered_df_lang = df_language[(df_language['Date'] >= start_date_lang) & (df_language['Date'] <= end_date_lang)]

# Chart type selection for language model metrics
chart_type_lang = st.sidebar.selectbox("Select Chart Type", ["line", "bar", "scatter"])

# Display the selected language model metric chart
st.subheader(f"Chart for {selected_language_metric}")
fig_lang = px.scatter(filtered_df_lang, x='Date', y=selected_language_metric, title=f"{selected_language_metric} Over Time")
fig_lang.update_traces(hovertemplate="Date: %{x}<br>Value: %{y}")
st.plotly_chart(fig_lang, use_container_width=True)

# Interactive Elements for language model metrics
st.subheader("Interactive Elements")
st.write("Hover over the chart to see details. You can customize the chart on the sidebar.")

# Error Handling for language model metrics
if selected_language_metric not in df_language.columns:
    st.error("Selected language metric not found in the dataset. Please choose a valid metric for the language model.")

# Image Model Metrics Section
st.header("Image Model Metrics")
st.subheader("Data Summary")
st.write("This section visualizes various metrics related to the performance of an image model over time.")

# Sidebar for image model metric selection
selected_image_metric = st.sidebar.selectbox("Select Image Metric", df_image_metrics.columns)

# Date range selection for image model metrics
start_date_img = st.sidebar.date_input("Start Date", min(df_image_metrics['Date']), max(df_image_metrics['Date']))
end_date_img = st.sidebar.date_input("End Date", min(df_image_metrics['Date']), max(df_image_metrics['Date']))
filtered_df_img = df_image_metrics[(df_image_metrics['Date'] >= start_date_img) & (df_image_metrics['Date'] <= end_date_img)]

# Chart type selection for image model metrics
chart_type_img = st.sidebar.selectbox("Select Chart Type", ["line", "bar", "scatter"])

# Display the selected image model metric chart
st.subheader(f"Chart for {selected_image_metric}")
fig_img = px.scatter(filtered_df_img, x='Date', y=selected_image_metric, title=f"{selected_image_metric} Over Time")
fig_img.update_traces(hovertemplate="Date: %{x}<br>Value: %{y}")

# Visualization Customization for image model metrics
st.sidebar.subheader("Chart Customization for Image Model Metrics")
color_option_img = st.sidebar.checkbox("Use Custom Colors")
if color_option_img:
    color_img = st.sidebar.color_picker("Select Color", value='blue')
    fig_img.update_traces(marker=dict(color=color_img))

# Display the chart with or without customization for image model metrics
st.plotly_chart(fig_img, use_container_width=True)

# Additional Details for Specific Image Metrics
if selected_image_metric == 'Confusion Matrix':
    st.subheader("Confusion Matrix")
    st.write("A confusion matrix is a table that summarizes the model's performance, breaking down true positives, true negatives, false positives, and false negatives.")

elif selected_image_metric == 'IoU':
    st.subheader("Intersection over Union (IoU)")
    st.write("IoU measures the overlap between the predicted and ground truth bounding boxes or segmentation masks.")

elif selected_image_metric in ['Top-1 Accuracy', 'Top-5 Accuracy']:
    st.subheader(selected_image_metric)
    st.write(f"{selected_image_metric} evaluates the model's ability to correctly predict the top-most likely class and the top 5 most likely classes, respectively.")

elif selected_image_metric == 'Mean Average Precision (mAP)':
    st.subheader("Mean Average Precision (mAP)")
    st.write("mAP is an average of precision values calculated at different recall levels, commonly used in object detection tasks.")

elif selected_image_metric == 'Perceptual Metrics':
    st.subheader("Perceptual Metrics")
    st.write("Perceptual metrics like Structural Similarity Index (SSI) or Peak Signal-to-Noise Ratio (PSNR) measure the perceptual quality of images.")

elif selected_image_metric == 'Robustness':
    st.subheader("Robustness")
    st.write("Robustness measures the model's performance under different conditions such as variations in lighting, orientation, or noise.")

elif selected_image_metric == 'Latency (ms)':
    st.subheader("Latency")
    st.write("Latency is the time it takes for the model to process and generate predictions for an image.")

# Error Handling for image model metrics
if selected_image_metric not in df_image_metrics.columns:
    st.error("Selected image metric not found in the dataset. Please choose a valid metric for the image model.")
