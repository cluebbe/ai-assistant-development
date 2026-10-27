"""Optional Task 12: offer the repair lookup as an MCP server.

    python mcp_server.py

Any MCP client (an AI app or an AI code editor) can then use the tool,
without knowing anything about this code.
"""

import gradio as gr

from repair_tools import get_repair_status

demo = gr.Interface(
    fn=get_repair_status,  # Name, docstring and type hints become the tool description
    inputs=gr.Textbox(label="ticket_id", placeholder="KW-1042"),
    outputs=gr.JSON(label="Status"),
    title="Kurbelwerk repair status",
    api_name="get_repair_status",
)

if __name__ == "__main__":
    demo.launch(mcp_server=True)
