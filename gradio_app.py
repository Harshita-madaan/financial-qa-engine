"""
gradio_app.py — Gradio chat UI for Financial Report Q&A Engine.
Run: python gradio_app.py
"""
import os
import json
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Pre-load chain at startup
print("Loading knowledge base...")
chain = None
try:
    from src.qa_chain import build_chain
    chain = build_chain()
    print("✅ Knowledge base loaded!")
except Exception as e:
    print(f"❌ Error loading chain: {e}")


def answer_question(message, history):
    global chain
    history = history or []

    if chain is None:
        history.append({"role": "assistant", "content": "❌ Knowledge base not loaded. Restart the app."})
        return history, ""

    if not message.strip():
        return history, ""

    try:
        from src.qa_chain import ask
        result = ask(chain, message)

        response = result["answer"]

        if result["sources"]:
            response += "\n\n---\n📄 **Sources:**\n"
            for i, src in enumerate(result["sources"], 1):
                response += (
                    f"\n**[{i}]** {src.get('ticker','?')} · "
                    f"{src.get('filing_type','?')} · "
                    f"Page {src.get('page','?')}"
                )

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history, ""

    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ Error: {e}"})
        return history, ""


def get_index_info():
    index_dir = Path("data/index")
    if (index_dir / "metadata.json").exists():
        with open(index_dir / "metadata.json") as f:
            metas = json.load(f)
        tickers = sorted({m["ticker"] for m in metas if m.get("ticker")})
        filings = sorted({m["filing_type"] for m in metas if m.get("filing_type")})
        return (f"📊 **Indexed Data**\n\n"
                f"Tickers: {', '.join(tickers)}\n\n"
                f"Filing types: {', '.join(filings)}\n\n"
                f"Total chunks: {len(metas):,}\n\n"
                f"Status: {'✅ Ready' if chain else '❌ Not loaded'}")
    return "⚠️ No index found."


with gr.Blocks(title="Financial Report Q&A") as demo:

    gr.HTML("""
        <div style="text-align:center; padding:20px;">
            <h1>📊 Financial Report Q&A Engine</h1>
            <p style="color:#666;">Ask questions about real SEC filings from Apple, Microsoft & Google</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Setup")
            gr.Markdown(get_index_info())

            gr.Markdown("### 💡 Example Questions")
            gr.Markdown("""
- What was Apple's total revenue in FY2023?
- What are Microsoft's main risk factors?
- How did Google's operating income change year over year?
- What is Apple's gross margin percentage?
- What does Microsoft say about cloud growth?
            """)

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with the Filings")
            chatbot = gr.Chatbot(
                height=500,
                render_markdown=True,
            )
            msg = gr.Textbox(
                placeholder="Ask anything about the SEC filings...",
                label="Your question",
            )
            send_btn = gr.Button("Send ↗", variant="primary")
            clear = gr.Button("🗑️ Clear Chat")

            send_btn.click(
                answer_question,
                [msg, chatbot],
                [chatbot, msg]
            )
            msg.submit(
                answer_question,
                [msg, chatbot],
                [chatbot, msg]
            )
            clear.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
    )