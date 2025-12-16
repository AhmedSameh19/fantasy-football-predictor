"""
Visualization Module
Creates interactive graph visualizations of knowledge graph data

Uses NetworkX for graph structure and Plotly for interactive visualization.
"""

import networkx as nx
import plotly.graph_objects as go


def create_knowledge_graph_visualization(baseline_context):
    """
    Create an interactive graph visualization from baseline context.

    Extracts nodes (players, teams, positions) and edges (relationships)
    from the normalized baseline results and renders them as an interactive
    network graph using Plotly.

    Args:
        baseline_context: Normalized baseline results from retrieval

    Returns:
        plotly.graph_objects.Figure: Interactive graph visualization
    """
    G = nx.Graph()

    # Extract nodes and edges from baseline context
    for record in baseline_context:
        # Add player nodes (blue)
        for player in record.get("players", []):
            player_name = player.get("player_name", "Unknown")
            G.add_node(player_name, node_type="player", color="#3498db")

        # Add team nodes (red)
        for team in record.get("teams", []):
            team_name = team.get("team_name", "Unknown")
            G.add_node(team_name, node_type="team", color="#e74c3c")

        # Add position nodes (green)
        for position in record.get("positions", []):
            pos_name = position.get("position_name", "Unknown")
            G.add_node(pos_name, node_type="position", color="#2ecc71")

        # Add season nodes (purple)
        for season in record.get("seasons", []):
            season_name = season.get("season_name", "Unknown")
            G.add_node(season_name, node_type="season", color="#9b59b6")

        # Add gameweek nodes (orange)
        for gameweek in record.get("gameweeks", []):
            gw_number = gameweek.get("gameweek_number", "Unknown")
            gw_label = f"GW {gw_number}"
            G.add_node(gw_label, node_type="gameweek", color="#f39c12")

        # Add fixture nodes (teal)
        for fixture in record.get("fixtures", []):
            fixture_num = fixture.get("fixture_number", "Unknown")
            fixture_label = f"Fixture {fixture_num}"
            G.add_node(fixture_label, node_type="fixture", color="#1abc9c")

        # Add relationships as edges
        for rel in record.get("relationships", []):
            from_node_data = rel.get("from", "")
            to_node_data = rel.get("to", "")
            rel_type = rel.get("type", "")

            # Extract node names and add nodes if not already present
            from_node = None
            to_node = None

            # Handle "from" node
            if isinstance(from_node_data, dict):
                # Try to get a meaningful name from the node based on its type
                if "player_name" in from_node_data:
                    from_node = from_node_data["player_name"]
                    node_type = "player"
                    node_color = "#3498db"
                elif "fixture_number" in from_node_data:
                    from_node = f"Fixture {from_node_data['fixture_number']}"
                    node_type = "fixture"
                    node_color = "#1abc9c"
                elif "number" in from_node_data:
                    from_node = f"GW {from_node_data['number']}"
                    node_type = "gameweek"
                    node_color = "#f39c12"
                elif "name" in from_node_data:
                    from_node = from_node_data["name"]
                    # Determine type based on context
                    node_type = "other"
                    node_color = "#95a5a6"
                else:
                    from_node = "Unknown"
                    node_type = "other"
                    node_color = "#95a5a6"

                # Add node to graph if not already present
                if from_node and from_node not in G:
                    G.add_node(from_node, node_type=node_type, color=node_color)
            else:
                from_node = str(from_node_data) if from_node_data else None

            # Handle "to" node
            if isinstance(to_node_data, dict):
                # Try to get a meaningful name from the node based on its type
                if "player_name" in to_node_data:
                    to_node = to_node_data["player_name"]
                    node_type = "player"
                    node_color = "#3498db"
                elif "fixture_number" in to_node_data:
                    to_node = f"Fixture {to_node_data['fixture_number']}"
                    node_type = "fixture"
                    node_color = "#1abc9c"
                elif "number" in to_node_data:
                    to_node = f"GW {to_node_data['number']}"
                    node_type = "gameweek"
                    node_color = "#f39c12"
                elif "name" in to_node_data:
                    to_node = to_node_data["name"]
                    # Determine type based on context
                    node_type = "other"
                    node_color = "#95a5a6"
                else:
                    to_node = "Unknown"
                    node_type = "other"
                    node_color = "#95a5a6"

                # Add node to graph if not already present
                if to_node and to_node not in G:
                    G.add_node(to_node, node_type=node_type, color=node_color)
            else:
                to_node = str(to_node_data) if to_node_data else None

            # Add edge if both nodes are valid
            if from_node and to_node:
                G.add_edge(from_node, to_node, relationship=rel_type)

        # Create edges from aggregated stats (e.g., player comparisons)
        players = [p.get("player_name") for p in record.get("players", [])]
        if len(players) >= 2:
            # Connect compared players
            G.add_edge(players[0], players[1], relationship="compared_with")

    # Handle empty graph case
    if len(G.nodes()) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No graph data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        return fig

    # Create layout using spring layout for nice spacing
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Create edge traces (lines connecting nodes)
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
        )

    # Create node traces grouped by type (for legend)
    node_traces = {}
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 'unknown')
        color = G.nodes[node].get('color', '#999')

        if node_type not in node_traces:
            node_traces[node_type] = {
                'x': [], 'y': [], 'text': [],
                'color': color, 'name': node_type.capitalize()
            }

        x, y = pos[node]
        node_traces[node_type]['x'].append(x)
        node_traces[node_type]['y'].append(y)
        node_traces[node_type]['text'].append(node)

    # Create Plotly figure
    fig = go.Figure()

    # Add edges
    for trace in edge_trace:
        fig.add_trace(trace)

    # Add nodes
    for node_type, data in node_traces.items():
        fig.add_trace(go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='markers+text',
            name=data['name'],
            text=data['text'],
            textposition="top center",
            marker=dict(
                size=20,
                color=data['color'],
                line=dict(width=2, color='white')
            ),
            hoverinfo='text'
        ))

    # Update layout
    fig.update_layout(
        title="Knowledge Graph Visualization",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        plot_bgcolor='rgba(250,250,250,0.9)'
    )

    return fig
