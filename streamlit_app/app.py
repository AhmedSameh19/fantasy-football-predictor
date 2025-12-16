"""
Fantasy Premier League Graph-RAG Assistant
Main Streamlit Application

This is the main entry point for the application.
Provides an interactive UI for querying the FPL knowledge graph.
"""

import streamlit as st
import pandas as pd
import concurrent.futures

# Import application modules
from config import EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL
from database import get_kg_entities
from retrieval import get_context_enhanced
from llm_service import openai_generate, gemini_generate, openrouter_generate
from visualization import create_knowledge_graph_visualization

# Page configuration
st.set_page_config(
    page_title="Fantasy Premier League Assistant",
    page_icon="⚽",
    layout="wide"
)

# Main title
st.title("⚽ Fantasy Premier League Graph-RAG Assistant")
st.markdown("Ask questions about players, teams, fixtures, and more!")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Retrieval method selection
    retrieval_method = st.selectbox(
        "Retrieval Method",
        ["both", "baseline", "embeddings"],
        help="Choose how to retrieve context from the knowledge graph"
    )

    # Embedding model selection
    embedding_model = st.selectbox(
        "Embedding Model",
        list(EMBEDDING_MODELS.keys()),
        index=list(EMBEDDING_MODELS.keys()).index(DEFAULT_EMBEDDING_MODEL),
        help="Choose which embedding model to use for semantic search"
    )

    # Display embedding model info
    model_info = EMBEDDING_MODELS[embedding_model]
    st.caption(f"📊 {model_info['name']}")
    st.caption(f"Dimensions: {model_info['dimension']}")

    # Model comparison mode
    compare_models = st.checkbox(
        "Compare All Models",
        value=False,
        help="Run all 3 models simultaneously for comparison"
    )

    if not compare_models:
        selected_model = st.selectbox(
            "Select Model",
            ['Gemini 2.5 Flash', 'LLama 3.3 (Openrouter)', 'OpenAI GPT 5.1']
        )

    st.divider()

    st.header("ℹ️ About")
    st.markdown("""
    **Retrieval Methods:**
    - **Baseline**: Cypher queries only
    - **Embeddings**: Semantic similarity only
    - **Both**: Hybrid approach (recommended)

    **Embedding Models:**
    - **all-MiniLM-L6-v2**: Faster, 384 dimensions
    - **all-mpnet-base-v2**: More accurate, 768 dimensions

    **Features:**
    - Natural language queries
    - Knowledge graph retrieval
    - Multiple LLM options
    - Real-time model comparison
    - Graph visualization
    """)

    st.divider()

    st.header("📊 Database Info")
    kg_entities = get_kg_entities()
    st.metric("Players", len(kg_entities['players']))
    st.metric("Teams", len(kg_entities['teams']))
    st.metric("Positions", len(kg_entities['positions']))
    st.metric("Seasons", len(kg_entities['seasons']))

# Main input
user_query = st.text_input(
    "Ask a question:",
    placeholder="e.g., Who performed better in GW15, Salah or Haaland?"
)

# Submit button
if st.button("Get Answer", type="primary"):
    if not user_query:
        st.error("Please enter a question!")
    else:
        with st.spinner("🔍 Processing your question..."):
            try:
                # Get context with enhanced function
                text_context, baseline_context, embeddings_context, cypher_query, query_params = get_context_enhanced(
                    user_query,
                    retrieval_method=retrieval_method,
                    embedding_model=embedding_model
                )

                # Display Cypher Query (only for baseline or both)
                if retrieval_method in ["baseline", "both"]:
                    st.subheader("🔍 Cypher Query Executed")
                    with st.expander("View Query Details", expanded=False):
                        if cypher_query:
                            st.code(cypher_query, language="cypher")
                            if query_params:
                                st.write("**Parameters:**")
                                st.json(query_params)
                        else:
                            st.info("No Cypher query was generated")

                # Display Graph Visualization (only for baseline or both)
                if retrieval_method in ["baseline", "both"]:
                    st.subheader("📊 Knowledge Graph Visualization")
                    if baseline_context:
                        fig = create_knowledge_graph_visualization(baseline_context)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No graph data to visualize")

                    # Display retrieved baseline data
                    with st.expander("📊 Knowledge Graph Data Retrieved", expanded=False):
                        if baseline_context:
                            for i, record in enumerate(baseline_context, start=1):
                                st.markdown(f"**Result {i}**")

                                # Display all fields in the record dynamically
                                for key, value in record.items():
                                    if value:  # Only show non-empty fields
                                        # Format the key nicely
                                        formatted_key = key.replace("_", " ").title()

                                        # Use appropriate emoji based on field name
                                        if "player" in key.lower():
                                            emoji = "🧑"
                                        elif "team" in key.lower():
                                            emoji = "🏟"
                                        elif "stat" in key.lower() or "point" in key.lower():
                                            emoji = "📈"
                                        elif "fixture" in key.lower() or "match" in key.lower():
                                            emoji = "⚽"
                                        elif "season" in key.lower():
                                            emoji = "📅"
                                        elif "gameweek" in key.lower() or "gw" in key.lower():
                                            emoji = "🎯"
                                        elif "position" in key.lower():
                                            emoji = "📍"
                                        else:
                                            emoji = "📊"

                                        st.markdown(f"**{emoji} {formatted_key}**")

                                        # Try to display as JSON, fall back to write if it fails
                                        try:
                                            # If it's already a dict or list, use st.json
                                            if isinstance(value, (dict, list)):
                                                st.json(value)
                                            else:
                                                # For other types, convert to string and use st.write
                                                st.write(value)
                                        except Exception as e:
                                            # If all else fails, display as text
                                            st.text(str(value))
                        else:
                            st.info("No baseline data retrieved")

                # Display Embeddings Results (only for embeddings or both)
                if retrieval_method in ["embeddings", "both"]:
                    st.subheader("🔮 Semantic Similarity Results")
                    if embeddings_context:
                        with st.expander("View Similar Players", expanded=True):
                            # Display each embedding result with player info and node properties
                            for i, (player, score, node) in enumerate(embeddings_context, start=1):
                                st.markdown(f"**Result {i}: {player}** (Similarity: {score:.4f})")

                                # Convert node to dict if needed
                                node_dict = dict(node) if hasattr(node, '__iter__') and not isinstance(node, dict) else node

                                # Display node properties
                                if node_dict:
                                    st.json(node_dict)
                                else:
                                    st.info("No node properties available")

                                if i < len(embeddings_context):
                                    st.divider()

                            st.caption(f"Using model: {embedding_model}")
                    else:
                        st.info("No embedding-based results found")

                st.divider()

                # Model comparison or single model
                if compare_models:
                    st.subheader("🤖 Model Comparison")

                    # Run all models in parallel
                    models = {
                        'Gemini 2.5 Flash': gemini_generate,
                        'LLama 3.3 (Openrouter)': openrouter_generate,
                        'OpenAI GPT 5.1': openai_generate
                    }

                    results = {}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        future_to_model = {
                            executor.submit(func, str(text_context), user_query): name
                            for name, func in models.items()
                        }

                        for future in concurrent.futures.as_completed(future_to_model):
                            model_name = future_to_model[future]
                            try:
                                results[model_name] = future.result()
                            except Exception as e:
                                st.error(f"Error with {model_name}: {str(e)}")

                    # Display results in columns
                    cols = st.columns(3)

                    for idx, (model_name, result) in enumerate(results.items()):
                        with cols[idx]:
                            st.markdown(f"### {model_name}")
                            st.info(result["response"])

                            st.markdown("**Metrics:**")
                            st.metric("Time", f"{result['metrics']['response_time']}s")
                            st.metric("Tokens", result['metrics']['token_usage']['total_tokens'])
                            st.metric("Cost", f"${result['metrics']['cost']}")

                    # Comparison table
                    st.subheader("📊 Metrics Comparison")
                    comparison_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'Response Time (s)': [r['metrics']['response_time'] for r in results.values()],
                        'Total Tokens': [r['metrics']['token_usage']['total_tokens'] for r in results.values()],
                        'Cost ($)': [r['metrics']['cost'] for r in results.values()]
                    })
                    st.dataframe(comparison_df, use_container_width=True)

                else:
                    # Single model mode
                    st.subheader("🤖 LLM Answer")

                    if selected_model == 'Gemini 2.5 Flash':
                        result = gemini_generate(str(text_context), user_query)
                    elif selected_model == 'LLama 3.3 (Openrouter)':
                        result = openrouter_generate(str(text_context), user_query)
                    else:
                        result = openai_generate(str(text_context), user_query)

                    st.info(result["response"])

                    st.subheader("📊 Metrics")
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("Response Time", f"{result['metrics']['response_time']}s")
                    with col2:
                        st.metric("Input Tokens", result['metrics']['token_usage']['prompt_tokens'])
                    with col3:
                        st.metric("Output Tokens", result['metrics']['token_usage']['completion_tokens'])
                    with col4:
                        st.metric("Total Tokens", result['metrics']['token_usage']['total_tokens'])
                    with col5:
                        st.metric("Cost", f"${result['metrics']['cost']}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
