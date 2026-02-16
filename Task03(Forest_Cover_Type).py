import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Forest Cover Type Classification",
    page_icon="🌲",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #2e5c3e, #1e3c2e);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2e5c3e, #1e3c2e);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🌲 Forest Cover Type Classification</h1>
        <p style='font-size: 1.2rem;'>Backend Training | Frontend Display | Real-time Predictions</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state['models_trained'] = False
if 'results' not in st.session_state:
    st.session_state['results'] = {}
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'feature_names' not in st.session_state:
    st.session_state['feature_names'] = None
if 'feature_means' not in st.session_state:
    st.session_state['feature_means'] = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/forest.png", width=80)
    st.title("🎮 Control Panel")
    
    # Train button
    if st.button(" Train Models (Backend)", use_container_width=True):
        st.session_state['train_clicked'] = True
    
    st.markdown("---")
    
    # Model info
    st.subheader("📊 Model Status")
    if st.session_state['models_trained']:
        st.success("✅ Models are trained and ready!")
    else:
        st.warning(" Models not trained yet")
    
    st.markdown("---")
    
    # Dataset info
    st.subheader(" Dataset Info")
    st.info("""
    **covtype.csv**
    - Samples: 581,012
    - Features: 54
    - Classes: 7
    """)

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Data Overview", 
    " Model Results", 
    " Feature Importance",
    " Make Predictions",
    " Comparison"
])

# Check if training is requested
if 'train_clicked' in st.session_state and st.session_state['train_clicked']:
    with st.spinner(" Training models in backend... Please wait..."):
        
        # --- BACKEND TRAINING ---
        # Load data
        df = pd.read_csv('covtype.csv')
        
        # Preprocess
        X = df.drop('Cover_Type', axis=1)
        y = df['Cover_Type'] - 1
        
        # Store feature names and means for later use
        st.session_state['feature_names'] = X.columns.tolist()
        st.session_state['feature_means'] = X.mean().to_dict()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        
        results = {}
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        
        results['Random Forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'accuracy': rf_acc,
            'report': classification_report(y_test, rf_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, rf_pred)
        }
        
        # Train XGBoost
        xgb_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        
        results['XGBoost'] = {
            'model': xgb_model,
            'predictions': xgb_pred,
            'accuracy': xgb_acc,
            'report': classification_report(y_test, xgb_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, xgb_pred)
        }
        
        # Store results
        st.session_state['results'] = results
        st.session_state['models_trained'] = True
        st.session_state['train_clicked'] = False
        
        st.success("✅ Models trained successfully! Check the tabs for results.")
        st.balloons()
        time.sleep(2)
        st.rerun()

# Tab 1: Data Overview
with tab1:
    if st.session_state['models_trained']:
        st.header("📊 Dataset Overview")
        
        # Load data for display
        df = pd.read_csv('covtype.csv')
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", f"{df.shape[0]:,}")
        with col2:
            st.metric("Features", df.shape[1] - 1)
        with col3:
            st.metric("Cover Types", df['Cover_Type'].nunique())
        with col4:
            st.metric("Test Samples", f"{len(st.session_state['X_test']):,}")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        # Distribution
        st.subheader("Cover Type Distribution")
        cover_dist = df['Cover_Type'].value_counts().sort_index()
        fig_dist = px.bar(
            x=cover_dist.index,
            y=cover_dist.values,
            text=cover_dist.values,
            title="Distribution of Forest Cover Types",
            labels={'x': 'Cover Type', 'y': 'Count'},
            color=cover_dist.values,
            color_continuous_scale='greens'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("👆 Click 'Train Models' in sidebar to load data and train models")

# Tab 2: Model Results
with tab2:
    if st.session_state['models_trained']:
        st.header("📈 Model Performance Results")
        
        results = st.session_state['results']
        
        # Accuracy comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>🌳 Random Forest</h3>
                    <h2 style='color: #2e5c3e;'>{results['Random Forest']['accuracy']:.4f}</h2>
                    <p>Accuracy</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>⚡ XGBoost</h3>
                    <h2 style='color: #1e3c2e;'>{results['XGBoost']['accuracy']:.4f}</h2>
                    <p>Accuracy</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Confusion Matrices
        st.subheader("Confusion Matrices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🌳 Random Forest**")
            fig_cm1, ax_cm1 = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                results['Random Forest']['confusion_matrix'], 
                annot=True, fmt='d', cmap='Greens', ax=ax_cm1
            )
            ax_cm1.set_xlabel('Predicted')
            ax_cm1.set_ylabel('Actual')
            st.pyplot(fig_cm1)
        
        with col2:
            st.markdown("**⚡ XGBoost**")
            fig_cm2, ax_cm2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                results['XGBoost']['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues', ax=ax_cm2
            )
            ax_cm2.set_xlabel('Predicted')
            ax_cm2.set_ylabel('Actual')
            st.pyplot(fig_cm2)
        
        # Classification Reports
        st.subheader("Classification Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🌳 Random Forest**")
            rf_report = pd.DataFrame(results['Random Forest']['report']).transpose()
            st.dataframe(rf_report.style.highlight_max(axis=0))
        
        with col2:
            st.markdown("**⚡ XGBoost**")
            xgb_report = pd.DataFrame(results['XGBoost']['report']).transpose()
            st.dataframe(xgb_report.style.highlight_max(axis=0))
    else:
        st.info("👆 Train models first to see results")

# Tab 3: Feature Importance
with tab3:
    if st.session_state['models_trained']:
        st.header(" Feature Importance Analysis")
        
        results = st.session_state['results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌳 Random Forest - Top 10 Features")
            rf_imp = pd.Series(
                results['Random Forest']['model'].feature_importances_,
                index=st.session_state['feature_names']
            ).sort_values(ascending=False).head(10)
            
            fig_rf = px.bar(
                x=rf_imp.values,
                y=rf_imp.index,
                orientation='h',
                title="Random Forest Feature Importance",
                labels={'x': 'Importance', 'y': 'Features'},
                color=rf_imp.values,
                color_continuous_scale='greens'
            )
            st.plotly_chart(fig_rf, use_container_width=True)
            
            # Table
            rf_imp_df = pd.DataFrame({
                'Feature': rf_imp.index,
                'Importance': rf_imp.values
            })
            st.dataframe(rf_imp_df)
        
        with col2:
            st.subheader("⚡ XGBoost - Top 10 Features")
            xgb_imp = pd.Series(
                results['XGBoost']['model'].feature_importances_,
                index=st.session_state['feature_names']
            ).sort_values(ascending=False).head(10)
            
            fig_xgb = px.bar(
                x=xgb_imp.values,
                y=xgb_imp.index,
                orientation='h',
                title="XGBoost Feature Importance",
                labels={'x': 'Importance', 'y': 'Features'},
                color=xgb_imp.values,
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig_xgb, use_container_width=True)
            
            # Table
            xgb_imp_df = pd.DataFrame({
                'Feature': xgb_imp.index,
                'Importance': xgb_imp.values
            })
            st.dataframe(xgb_imp_df)
    else:
        st.info("👆 Train models first to see feature importance")

# Tab 4: Make Predictions (FIXED VERSION)
with tab4:
    if st.session_state['models_trained']:
        st.header(" Make Predictions")
        
        st.markdown("""
        <div class="prediction-box">
            <h3>Enter feature values to predict forest cover type</h3>
            <p>Adjust the values below and click Predict</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create input fields for ALL features in correct order
        st.subheader("Enter Feature Values")
        
        # Get all feature names in the correct order (same as training)
        all_features = st.session_state['feature_names']
        
        # Create a dictionary to store input values
        input_values = {}
        
        # Use columns for better layout
        col1, col2, col3 = st.columns(3)
        
        # Distribute features across columns
        for idx, feature in enumerate(all_features):
            # Determine which column to use
            if idx % 3 == 0:
                col = col1
            elif idx % 3 == 1:
                col = col2
            else:
                col = col3
            
            # Default value from feature means
            default_val = st.session_state['feature_means'][feature]
            
            # Create input field
            with col:
                input_values[feature] = st.number_input(
                    f"**{feature}**",
                    value=float(default_val),
                    step=0.1,
                    format="%.2f",
                    key=f"pred_{feature}_{idx}"
                )
        
        # Predict button
        if st.button(" Predict Cover Type", use_container_width=True):
            
            # Create dataframe with features in EXACT same order as training
            input_df = pd.DataFrame([input_values])
            
            # Ensure columns are in the same order
            input_df = input_df[all_features]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Random Forest prediction
                rf_pred = st.session_state['results']['Random Forest']['model'].predict(input_df)[0] + 1
                rf_proba = st.session_state['results']['Random Forest']['model'].predict_proba(input_df).max()
                
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>🌳 Random Forest</h3>
                        <h1 style='color: #2e5c3e;'>Type {rf_pred}</h1>
                        <p>Confidence: {rf_proba:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # XGBoost prediction
                xgb_pred = st.session_state['results']['XGBoost']['model'].predict(input_df)[0] + 1
                xgb_proba = st.session_state['results']['XGBoost']['model'].predict_proba(input_df).max()
                
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>⚡ XGBoost</h3>
                        <h1 style='color: #1e3c2e;'>Type {xgb_pred}</h1>
                        <p>Confidence: {xgb_proba:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Cover type descriptions
            st.subheader(" Cover Type Descriptions")
            cover_types = {
                1: "Spruce/Fir",
                2: "Lodgepole Pine",
                3: "Ponderosa Pine",
                4: "Cottonwood/Willow",
                5: "Aspen",
                6: "Douglas-fir",
                7: "Krummholz"
            }
            
            # Create description dataframe
            desc_df = pd.DataFrame([
                {"Cover Type": k, "Description": v, "Forest Type": v.split('/')[0] if '/' in v else v}
                for k, v in cover_types.items()
            ])
            
            st.dataframe(desc_df, use_container_width=True)
            
            # Show prediction interpretation
            st.info(f"""
            **Interpretation:**
            - Both models predict **Type {rf_pred if rf_pred == xgb_pred else f'{rf_pred} and {xgb_pred}'}**
            - {'✅ Models agree!' if rf_pred == xgb_pred else '⚠️ Models disagree - consider ensemble approach'}
            """)
    else:
        st.info("👆 Train models first to make predictions")

# Tab 5: Comparison
with tab5:
    if st.session_state['models_trained']:
        st.header(" Model Comparison")
        
        results = st.session_state['results']
        
        # Comparison dataframe
        comparison_data = {
            'Metric': ['Accuracy', 'Precision (macro avg)', 'Recall (macro avg)', 'F1-score (macro avg)'],
            'Random Forest': [
                f"{results['Random Forest']['accuracy']:.4f}",
                f"{results['Random Forest']['report']['macro avg']['precision']:.4f}",
                f"{results['Random Forest']['report']['macro avg']['recall']:.4f}",
                f"{results['Random Forest']['report']['macro avg']['f1-score']:.4f}"
            ],
            'XGBoost': [
                f"{results['XGBoost']['accuracy']:.4f}",
                f"{results['XGBoost']['report']['macro avg']['precision']:.4f}",
                f"{results['XGBoost']['report']['macro avg']['recall']:.4f}",
                f"{results['XGBoost']['report']['macro avg']['f1-score']:.4f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Style the dataframe
        st.dataframe(
            comparison_df.style.highlight_max(subset=['Random Forest', 'XGBoost'], axis=0),
            use_container_width=True
        )
        
        # Winner announcement
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate per-class performance
            rf_per_class = pd.DataFrame(results['Random Forest']['report']).transpose().iloc[:-3, :3]
            st.subheader("🌳 Random Forest - Per Class")
            st.dataframe(rf_per_class)
        
        with col2:
            xgb_per_class = pd.DataFrame(results['XGBoost']['report']).transpose().iloc[:-3, :3]
            st.subheader("⚡ XGBoost - Per Class")
            st.dataframe(xgb_per_class)
        
        # Winner
        if results['Random Forest']['accuracy'] > results['XGBoost']['accuracy']:
            st.success(" **Random Forest** performs better on this dataset!")
        elif results['XGBoost']['accuracy'] > results['Random Forest']['accuracy']:
            st.success(" **XGBoost** performs better on this dataset!")
        else:
            st.info(" Both models have similar performance")
    else:
        st.info("👆 Train models first to see comparison")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>🌲 Forest Cover Type Classification | Backend Training | Frontend Display | Real-time Predictions</p>
        <p style='font-size: 12px;'>Built with Python, Scikit-learn, XGBoost & Streamlit</p>
    </div>
""", unsafe_allow_html=True)