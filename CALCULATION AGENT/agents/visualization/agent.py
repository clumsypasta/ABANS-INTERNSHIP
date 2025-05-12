import json
import base64
from typing import Dict, List, Any
import io

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool


def plot_positions(positions_json: str) -> Dict[str, Any]:
    """
    Generate visualizations for position data.
    
    Args:
        positions_json: JSON string containing position data
        
    Returns:
        Dictionary containing visualization data
    """
    try:
        # Parse positions from JSON
        positions_data = json.loads(positions_json)
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(positions_data)
        
        # Create visualizations
        figures = {}
        
        # 1. Bar chart of open quantities per contract
        fig_qty = px.bar(
            df, 
            x='contract', 
            y='open_qty',
            title='Open Quantities by Contract',
            labels={'contract': 'Contract', 'open_qty': 'Open Quantity'},
            color='contract'
        )
        
        # 2. Comparison of FIFO vs LIFO WAP
        fig_wap = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_wap.add_trace(
            go.Bar(
                x=df['contract'],
                y=df['fifo_wap'],
                name='FIFO WAP',
                marker_color='blue'
            )
        )
        
        fig_wap.add_trace(
            go.Bar(
                x=df['contract'],
                y=df['lifo_wap'],
                name='LIFO WAP',
                marker_color='green'
            )
        )
        
        fig_wap.update_layout(
            title='FIFO vs LIFO Weighted Average Prices',
            xaxis_title='Contract',
            yaxis_title='Price'
        )
        
        # 3. Table of position data
        fig_table = go.Figure(data=[go.Table(
            header=dict(
                values=['Contract', 'Open Qty', 'FIFO WAP', 'LIFO WAP', 'Exchange', 'Expiry', 'Client Code', 'Strategy', 'Code', 'Tagging'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    df['contract'],
                    df['open_qty'],
                    df['fifo_wap'].round(2),
                    df['lifo_wap'].round(2),
                    df['exchange'],
                    df['expiry'],
                    df['client_code'],
                    df['strategy'],
                    df['code'],
                    df['tagging']
                ],
                fill_color='lavender',
                align='left'
            )
        )])
        
        fig_table.update_layout(title='Position Data Table')
        
        # Convert figures to JSON
        figures['qty_chart'] = fig_qty.to_json()
        figures['wap_chart'] = fig_wap.to_json()
        figures['table'] = fig_table.to_json()
        
        # Create HTML representation
        html_buffer = io.StringIO()
        html_buffer.write("<html><head><title>Position Analysis</title>")
        html_buffer.write("<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>")
        html_buffer.write("</head><body>")
        
        # Add table as HTML
        html_buffer.write("<h2>Position Data</h2>")
        html_buffer.write(df.to_html(index=False, classes='table table-striped'))
        
        # Add charts
        html_buffer.write("<div id='qty_chart'></div>")
        html_buffer.write("<div id='wap_chart'></div>")
        
        # Add JavaScript to render charts
        html_buffer.write("<script>")
        html_buffer.write(f"var qty_data = {fig_qty.to_json()};")
        html_buffer.write("Plotly.newPlot('qty_chart', qty_data.data, qty_data.layout);")
        html_buffer.write(f"var wap_data = {fig_wap.to_json()};")
        html_buffer.write("Plotly.newPlot('wap_chart', wap_data.data, wap_data.layout);")
        html_buffer.write("</script>")
        
        html_buffer.write("</body></html>")
        
        html_content = html_buffer.getvalue()
        
        return {
            "status": "success",
            "figures": figures,
            "html": html_content,
            "position_count": len(df)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Error generating visualizations: {str(e)}"
        }


# Create the Visualization Agent
visualization_agent = Agent(
    model="gemini-2.0-flash",
    name="visualization_agent",
    description="Generates charts and tables for position data",
    instruction="""
    You are a Visualization Agent that creates charts and tables.
    Your job is to generate visualizations for position data.
    Use the 'plot_positions' tool to create charts and tables for the given positions.
    """,
    tools=[plot_positions]
)
