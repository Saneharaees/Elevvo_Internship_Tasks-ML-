import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import time
import subprocess
import sys

# Try to import plotly, install if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not installed. Using matplotlib/seaborn for visualizations. For better visuals, install plotly with: pip install plotly")

# --- 1. PAGE CONFIG & ADVANCED THEME ---
st.set_page_config(
    page_title="EduPredict Pro | Student Performance Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    /* Main background and text */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom fonts and styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #00ffcc !important;
        background: linear-gradient(45deg, #00ffcc, #00b8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #a0aec0 !important;
    }
    
    /* Custom card design */
    .custom-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: transform 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
    }
    
    /* Prediction cards */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    
    .prediction-card h4 {
        color: white;
        font-size: 18px;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 15px;
    }
    
    .prediction-card h1 {
        font-size: 48px !important;
        font-weight: 800 !important;
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 50px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: none;
        padding: 15px 30px;
        font-size: 16px;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.75);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 7px 20px 0 rgba(102, 126, 234, 0.9);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1f2e 0%, #2d3748 100%);
    }
    
    .sidebar-content {
        padding: 20px;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00ffcc, #667eea);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 50px;
        padding: 10px 25px;
        font-weight: 600;
        color: #a0aec0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white !important;
    }
    
    /* Headers */
    .gradient-text {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 48px;
        text-align: center;
        margin-bottom: 30px;
    }
    
    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: #00ffcc;
        border-bottom: 2px solid #667eea;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Input container */
    .input-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 1s ease-out;
    }
    
    /* Installation box */
    .install-box {
        background: #1e2130;
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
    }
    
    .install-code {
        background: #2d3748;
        color: #00ffcc;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to install packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Installation option in sidebar if plotly is missing
if not PLOTLY_AVAILABLE:
    with st.sidebar:
        st.markdown("""
        <div class='install-box'>
            <h4 style='color: #ff6b6b;'>📦 Missing Package</h4>
            <p style='color: white;'>Plotly is required for interactive charts</p>
            <div class='install-code'>pip install plotly</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Install Plotly Now", use_container_width=True):
            with st.spinner("Installing plotly..."):
                try:
                    install_package("plotly")
                    st.success("✅ Plotly installed successfully! Please restart the app.")
                    st.cache_data.clear()
                except:
                    st.error("❌ Installation failed. Please run manually: pip install plotly")

# --- 2. DATA ENGINE WITH ENHANCEMENTS ---
@st.cache_data
def get_clean_data():
    try:
        df = pd.read_csv('StudentPerformanceFactors.csv')
        df = df.dropna()
        
        # Advanced feature engineering
        mappings = {'Low': 1, 'Medium': 2, 'High': 3}
        for col in ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level']:
            if col in df.columns:
                df[col] = df[col].map(mappings)
        
        # Create interaction features
        df['Study_Efficiency'] = df['Hours_Studied'] * (df['Attendance'] / 100)
        df['Sleep_Study_Ratio'] = df['Sleep_Hours'] / df['Hours_Studied'].replace(0, 0.1)
        df['Previous_Score_Impact'] = df['Previous_Scores'] * df['Hours_Studied']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- 3. MAIN APPLICATION ---
df = get_clean_data()

if df is not None:
    # Header with animation
    st.markdown("<h1 class='gradient-text fade-in'>📚 EduPredict Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0aec0; margin-bottom: 40px;'>Advanced Student Performance Analytics Platform</p>", unsafe_allow_html=True)
    
    # --- SIDEBAR WITH MINIMAL CONFIGURATION ---
    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        st.image("https://img.icons8.com/fluency/96/student-center.png", width=80)
        st.markdown("<h2 style='color: white; text-align: center;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Feature selection
        all_feats = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Parental_Involvement', 
                     'Access_to_Resources', 'Sleep_Hours', 'Tutoring_Sessions', 'Study_Efficiency',
                     'Sleep_Study_Ratio', 'Previous_Score_Impact']
        available_features = [f for f in all_feats if f in df.columns]
        
        st.markdown("<p style='color: #a0aec0;'>📊 Select Features</p>", unsafe_allow_html=True)
        selected_features = st.multiselect(
            "Choose predictive features",
            available_features,
            default=available_features[:5] if len(available_features) >= 5 else available_features,
            help="Select the features you want to use for prediction"
        )
        
        st.markdown("---")
        
        # Model selection
        st.markdown("<p style='color: #a0aec0;'>🧠 Model Selection</p>", unsafe_allow_html=True)
        use_linear = st.checkbox("Linear Regression", value=True)
        use_poly = st.checkbox("Polynomial Regression", value=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # --- MAIN CONTENT AREA ---
    if selected_features:
        # Quick stats row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Students", f"{len(df):,}")
        with col2:
            st.metric("Avg. Score", f"{df['Exam_Score'].mean():.1f}%")
        with col3:
            pass_rate = (df['Exam_Score'] > 60).mean() * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        
        st.markdown("---")
        
        # --- INPUT SECTION ON MAIN PAGE ---
        st.markdown("<h2 class='section-header'> Enter Student Details</h2>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='input-container'>", unsafe_allow_html=True)
            
            # Create input fields in rows of 3
            user_vals = []
            cols = st.columns(3)
            for idx, feat in enumerate(selected_features):
                with cols[idx % 3]:
                    val = st.number_input(
                        f"{feat.replace('_', ' ')}",
                        min_value=float(df[feat].min()),
                        max_value=float(df[feat].max()),
                        value=float(df[feat].median()),
                        step=0.1,
                        format="%.1f",
                        key=f"input_{feat}"
                    )
                    user_vals.append(val)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Prediction button
            predict_button = st.button(" PREDICT EXAM SCORE", use_container_width=True)
            
            # Progress animation
            if predict_button:
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
        
        st.markdown("---")
        
        # --- MODEL TRAINING ---
        X = df[selected_features]
        y = df['Exam_Score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # Linear Model
        if use_linear:
            lr = LinearRegression().fit(X_train_scaled, y_train)
            y_pred_lr = lr.predict(X_test_scaled)
            models['Linear'] = {'model': lr, 'predictions': y_pred_lr}
        
        # Polynomial Model
        if use_poly:
            poly_trans = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_train = poly_trans.fit_transform(X_train_scaled)
            X_poly_test = poly_trans.transform(X_test_scaled)
            poly_model = LinearRegression().fit(X_poly_train, y_train)
            y_pred_poly = poly_model.predict(X_poly_test)
            models['Polynomial'] = {'model': poly_model, 'predictions': y_pred_poly, 'transformer': poly_trans}

        # --- PREDICTION RESULTS ---
        if predict_button and user_vals and models:
            st.markdown("<h2 class='section-header'>📊 Prediction Results</h2>", unsafe_allow_html=True)
            
            # Prepare input
            input_data = np.array([user_vals])
            input_scaled = scaler.transform(input_data)
            
            # Create columns for predictions
            pred_cols = st.columns(len(models))
            
            for idx, (name, model_data) in enumerate(models.items()):
                with pred_cols[idx]:
                    if name == 'Linear':
                        pred = max(0, min(100, model_data['model'].predict(input_scaled)[0]))
                    else:
                        input_poly = model_data['transformer'].transform(input_scaled)
                        pred = max(0, min(100, model_data['model'].predict(input_poly)[0]))
                    
                    # Color based on prediction
                    color = '#00ffcc' if pred >= 60 else '#ff6b6b'
                    
                    st.markdown(f"""
                    <div class='prediction-card' style='background: linear-gradient(135deg, {color}40, {color}80);'>
                        <h4>{name} Model</h4>
                        <h1 style='color: white !important;'>{pred:.1f}%</h1>
                        <p style='color: white; margin-top: 10px; font-size: 16px;'>
                            {'✅ PASS' if pred >= 60 else '⚠️ NEEDS IMPROVEMENT'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show input summary
            with st.expander("📋 View Input Summary"):
                input_summary = pd.DataFrame({
                    'Feature': selected_features,
                    'Value': user_vals
                })
                st.dataframe(input_summary, use_container_width=True)
        
        st.markdown("---")
        
        # --- TABS FOR MODEL COMPARISON AND ANALYTICS ---
        tab1, tab2 = st.tabs(["📊 Model Comparison", "📈 Data Analytics"])
        
        with tab1:
            st.markdown("<h2 class='section-header'>Model Performance Comparison</h2>", unsafe_allow_html=True)
            
            if models:
                # Create comparison metrics
                metrics_data = []
                for name, model_data in models.items():
                    preds = model_data['predictions']
                    r2 = r2_score(y_test, preds)
                    mae = mean_absolute_error(y_test, preds)
                    metrics_data.append({
                        'Model': name,
                        'R² Score': r2,
                        'MAE': mae,
                        'Accuracy': r2 * 100
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                
                # Display metrics in columns
                metric_cols = st.columns(len(models))
                for idx, (name, model_data) in enumerate(models.items()):
                    with metric_cols[idx]:
                        r2 = r2_score(y_test, model_data['predictions'])
                        mae = mean_absolute_error(y_test, model_data['predictions'])
                        st.metric(f"{name} R²", f"{r2:.3f}")
                        st.metric(f"{name} MAE", f"{mae:.2f}")
                
                # Metrics visualization
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(data=[
                            go.Bar(name='R² Score', x=metrics_df['Model'], y=metrics_df['R² Score'], 
                                   marker_color='#00ffcc', text=metrics_df['R² Score'].round(3)),
                            go.Bar(name='MAE', x=metrics_df['Model'], y=metrics_df['MAE'], 
                                   marker_color='#ff6b6b', text=metrics_df['MAE'].round(2))
                        ])
                        fig.update_layout(title='Model Performance Metrics', barmode='group', 
                                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                        font_color='white')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        x = np.arange(len(metrics_df['Model']))
                        width = 0.35
                        ax.bar(x - width/2, metrics_df['R² Score'], width, label='R² Score', color='#00ffcc')
                        ax.bar(x + width/2, metrics_df['MAE'], width, label='MAE', color='#ff6b6b')
                        ax.set_xlabel('Model')
                        ax.set_ylabel('Score')
                        ax.set_title('Model Performance Metrics')
                        ax.set_xticks(x)
                        ax.set_xticklabels(metrics_df['Model'])
                        ax.legend()
                        ax.set_facecolor('#1e2130')
                        fig.patch.set_facecolor('#0e1117')
                        plt.setp(ax.spines.values(), color='white')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.title.set_color('white')
                        st.pyplot(fig)
                        plt.close()
                
                with col2:
                    # Feature importance for linear model
                    if use_linear:
                        importance = pd.Series(lr.coef_, index=selected_features).sort_values(ascending=True)
                        
                        if PLOTLY_AVAILABLE:
                            fig = go.Figure(data=[
                                go.Bar(x=importance.values, y=importance.index, orientation='h',
                                      marker_color='#667eea', text=importance.values.round(3))
                            ])
                            fig.update_layout(title='Feature Importance (Linear Model)', 
                                            xaxis_title='Coefficient Value',
                                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                            font_color='white')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            importance.plot(kind='barh', color='#667eea', ax=ax)
                            ax.set_title('Feature Importance (Linear Model)')
                            ax.set_xlabel('Coefficient Value')
                            ax.set_facecolor('#1e2130')
                            fig.patch.set_facecolor('#0e1117')
                            plt.setp(ax.spines.values(), color='white')
                            ax.tick_params(colors='white')
                            ax.xaxis.label.set_color('white')
                            ax.yaxis.label.set_color('white')
                            ax.title.set_color('white')
                            st.pyplot(fig)
                            plt.close()
                
                # Polynomial Regression Plot
                if use_poly:
                    st.markdown("<h3 style='color: #00ffcc; margin-top: 30px;'>Polynomial Regression Analysis</h3>", unsafe_allow_html=True)
                    
                    # Create scatter plot of actual vs predicted
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure()
                        
                        # Add scatter points
                        fig.add_trace(go.Scatter(
                            x=y_test, 
                            y=y_pred_poly,
                            mode='markers',
                            name='Predictions',
                            marker=dict(color='#667eea', size=8, opacity=0.6)
                        ))
                        
                        # Add perfect prediction line
                        fig.add_trace(go.Scatter(
                            x=[y_test.min(), y_test.max()],
                            y=[y_test.min(), y_test.max()],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='#00ffcc', width=2, dash='dash')
                        ))
                        
                        fig.update_layout(
                            title='Polynomial Regression: Actual vs Predicted',
                            xaxis_title='Actual Exam Score',
                            yaxis_title='Predicted Exam Score',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(y_test, y_pred_poly, alpha=0.6, color='#667eea', edgecolors='white', linewidth=0.5)
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                               '--', color='#00ffcc', linewidth=2, label='Perfect Prediction')
                        ax.set_xlabel('Actual Exam Score')
                        ax.set_ylabel('Predicted Exam Score')
                        ax.set_title('Polynomial Regression: Actual vs Predicted')
                        ax.legend()
                        ax.set_facecolor('#1e2130')
                        fig.patch.set_facecolor('#0e1117')
                        plt.setp(ax.spines.values(), color='white')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.title.set_color('white')
                        legend = ax.legend()
                        for text in legend.get_texts():
                            text.set_color('white')
                        st.pyplot(fig)
                        plt.close()
            else:
                st.info("Please select at least one model to compare")
        
        with tab2:
            st.markdown("<h2 class='section-header'>Data Analytics</h2>", unsafe_allow_html=True)
            
            # Correlation heatmap only
            corr_df = df[selected_features + ['Exam_Score']].corr()
            
            if PLOTLY_AVAILABLE:
                fig = go.Figure(data=go.Heatmap(
                    z=corr_df.values,
                    x=list(corr_df.columns),
                    y=list(corr_df.columns),
                    colorscale='Viridis',
                    text=corr_df.round(2).values,
                    texttemplate='%{text}',
                    textfont={"color": "white"}
                ))
                fig.update_layout(
                    title='Feature Correlation Matrix',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr_df, annot=True, cmap='viridis', ax=ax, cbar_kws={'label': 'Correlation'})
                ax.set_title('Feature Correlation Matrix')
                ax.set_facecolor('#1e2130')
                fig.patch.set_facecolor('#0e1117')
                plt.setp(ax.spines.values(), color='white')
                ax.tick_params(colors='white')
                ax.title.set_color('white')
                cbar = ax.collections[0].colorbar
                cbar.ax.yaxis.set_tick_params(color='white')
                cbar.ax.yaxis.label.set_color('white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                st.pyplot(fig)
                plt.close()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <p style='text-align: center; color: #a0aec0; font-size: 12px;'>
            EduPredict Pro | Student Performance Analytics Platform
        </p>
        """, unsafe_allow_html=True)
    
    else:
        st.info("👈 Please select at least one feature from the sidebar to begin analysis")

else:
    st.error("⚠️ File 'StudentPerformanceFactors.csv' not found. Please ensure the data file is in the correct directory.")
    
    # Upload option
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully! Please refresh the page.")