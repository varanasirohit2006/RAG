import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# --- Configuration & Styling ---
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for a professional "LinkedIn-ready" look
st.markdown("""
    <style>
    .main {
        background-color: var(--background-color);
    }
    .metric-container {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        text-align: center;
    }
    .stMetric {
        background-color: var(--secondary-background-color);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
    }
    .insight-card {
        background-color: rgba(59, 130, 246, 0.1);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 20px;
        border-top: 1px solid rgba(59, 130, 246, 0.15);
        border-right: 1px solid rgba(59, 130, 246, 0.15);
        border-bottom: 1px solid rgba(59, 130, 246, 0.15);
    }
    .insight-card h4, .insight-card p, .insight-card li, .insight-card ul {
        color: var(--text-color) !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 RAG Performance Evaluation")
st.markdown("### Benchmarking Retrieval Strategies for Multi-Modal RAG")
st.markdown("Comparing Vector, Keyword, Hybrid, and Reranking strategies across Ragas metrics.")

# --- Data Loading ---
CSV_PATH = "rag_performance_metrics.csv"

def load_data():
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            return df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None
    return None

df = load_data()

if df is not None:
    # Calculation of overall score
    metric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
    df['overall_score'] = df[metric_cols].mean(axis=1)
    
    # --- Top Row: Performance Highlights ---
    st.subheader("🏆 Performance Highlights")
    best_strategy = df.loc[df['overall_score'].idxmax()]
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Top Strategy", best_strategy['strategy'], help="Strategy with highest average score across all metrics")
    with m2:
        st.metric("Avg. Faithfulness", f"{df['faithfulness'].mean():.2f}", delta=f"{df['faithfulness'].max() - df['faithfulness'].mean():.2f} Max")
    with m3:
        st.metric("Avg. Answer Relevancy", f"{df['answer_relevancy'].mean():.2f}")
    with m4:
        st.metric("Overall System Score", f"{df['overall_score'].mean():.2f}")

    st.divider()

    # --- Middle Section: Comparative Visuals ---
    col_chart, col_table = st.columns([3, 2])
    
    with col_chart:
        st.subheader("📈 Metric Comparison by Strategy")
        
        fig = go.Figure()
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
        
        for i, metric in enumerate(metric_cols):
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=df['strategy'],
                y=df[metric],
                text=df[metric].round(2),
                textposition='auto',
                marker_color=colors[i % len(colors)]
            ))

        fig.update_layout(
            barmode='group',
            xaxis_title=None,
            yaxis_title="Score (0 to 1)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.subheader("📋 Raw Data Table")
        display_df = df.copy()
        # Sort by overall score
        display_df = display_df.sort_values(by='overall_score', ascending=False)
        
        # Display formatted table
        st.dataframe(
            display_df[['strategy'] + metric_cols].rename(columns=lambda x: x.replace('_', ' ').title()),
            use_container_width=True,
            hide_index=True,
            height=400
        )

    st.divider()

    # --- Bottom Section: LinkedIn Insights & Analysis ---
    st.subheader("💡 Analysis & Strategic Insights")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f"""
        <div class="insight-card">
        <h4>✅ Best Performing: {best_strategy['strategy']}</h4>
        <p>The <b>{best_strategy['strategy']}</b> strategy achieved the highest overall score of <b>{best_strategy['overall_score']:.2f}</b>. 
        It demonstrates superior performance in <b>{df.loc[df['strategy'] == best_strategy['strategy'], metric_cols].idxmax(axis=1).iloc[0].replace('_', ' ').title()}</b>.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="insight-card">
        <h4>🔍 Key Takeaways</h4>
        <ul>
            <li><b>Reranking</b> significantly boosts Context Precision by filtering out noise.</li>
            <li><b>Hybrid Search</b> balances the strengths of BM25 and Vector embeddings.</li>
            <li><b>Faithfulness</b> remains high across most strategies, indicating low hallucination.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.success("✨ **Tip for LinkedIn:** Use the 'Expand' button on the chart to get a full-screen view before taking your screenshot!")

    st.markdown("---")
    st.caption("RAG Performance Analytics Pipeline | Built with Streamlit, Plotly & Ragas")

else:
    st.warning("⚠️ No performance metrics found.")
    st.info("To populate this dashboard, please run the evaluation script from the main app or terminal:")
    st.code("python advanced_evaluation.py")
