import gradio as gr
from rag_chatbot import RAGChatbot

def create_gradio_interface(chatbot: RAGChatbot):
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# RAG Chatbot")
        gr.Markdown("Ask questions about the documents in the knowledge base.")

        chatbot_interface = gr.Chatbot(height=300)
        msg = gr.Textbox(placeholder="Ask a question", container=False, scale=7)
        context_docs = gr.JSON(value={}, label="Context Documents", container=False)
        session_id = gr.State()
        clear = gr.Button("Clear")
        delete_prev = gr.Button("Delete Previous")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history, session_id):
            bot_message, context, session_id = chatbot.chat(history[-1][0], history, session_id)
            history[-1][1] = bot_message
            return history, context, session_id

        def clear_session(session_id_str):
            new_session_id = chatbot.clear_session(session_id_str)
            return None, None, new_session_id

        def delete_previous(history, session_id):
            history = chatbot.delete_previous(history, session_id)
            return history, None

        msg.submit(user, [msg, chatbot_interface], [msg, chatbot_interface], queue=False).then(
            bot, [chatbot_interface, session_id], [chatbot_interface, context_docs, session_id]
        )
        clear.click(clear_session, [session_id], [chatbot_interface, context_docs, session_id], queue=False)
        delete_prev.click(delete_previous, [chatbot_interface, session_id], [chatbot_interface, context_docs], queue=False)

    return demo

if __name__ == "__main__":
    rag_chatbot = RAGChatbot()
    rag_chatbot.setup()

    # Uncomment the following line to save documents to the database
    # rag_chatbot.save_to_db()

    demo = create_gradio_interface(rag_chatbot)
    demo.launch(server_name="0.0.0.0")
