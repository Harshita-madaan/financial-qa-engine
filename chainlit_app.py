"""
chainlit_app.py — Chainlit chat UI for Financial Report Q&A Engine.
Run: chainlit run chainlit_app.py
"""
import os
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()


@cl.on_chat_start
async def start():
    await cl.Message(
        content="""👋 **Welcome to the Financial Report Q&A Engine!**

I can answer questions about real SEC filings (10-K / 10-Q) from:
- 🍎 **Apple (AAPL)**
- 🪟 **Microsoft (MSFT)**
- 🔍 **Google (GOOGL)**

Every answer comes with **cited sources** from the actual filings.

**Try asking:**
- What was Apple's total revenue in FY2023?
- What are Microsoft's main risk factors?
- How did Google's operating income change year over year?
        """
    ).send()

    # Build chain once and store in session
    await cl.Message(content="⏳ Loading knowledge base...").send()
    try:
        from src.qa_chain import build_chain
        chain = build_chain()
        cl.user_session.set("chain", chain)
        await cl.Message(content="✅ Knowledge base loaded! Ask me anything.").send()
    except Exception as e:
        await cl.Message(
            content=f"❌ Error loading index: {e}\n\nMake sure you have run:\n```\npython -m src.downloader\npython -m src.parser\npython -m src.indexer\n```"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    if not chain:
        await cl.Message(content="❌ Knowledge base not loaded. Please restart.").send()
        return

    # Show thinking indicator
    async with cl.Step(name="Searching filings...") as step:
        try:
            from src.qa_chain import ask
            result = ask(chain, message.content)

            step.output = f"Found {len(result['sources'])} relevant chunks"

        except Exception as e:
            await cl.Message(content=f"❌ Error: {e}").send()
            return

    # Send main answer
    await cl.Message(content=result["answer"]).send()

    # Send sources as elements
    if result["sources"]:
        elements = []
        for i, src in enumerate(result["sources"], 1):
            text = src.get("text", "")[:500] + "..."
            elements.append(
                cl.Text(
                    name=f"Source {i}: {src.get('ticker','?')} · {src.get('filing_type','?')} · Page {src.get('page','?')}",
                    content=text,
                    display="side",
                )
            )

        await cl.Message(
            content="📄 **Sources used:**",
            elements=elements,
        ).send()