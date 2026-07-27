import gradio as gr
from recommendation_engine import recommend_books, get_categories, get_tones

categories = get_categories()
tones = get_tones()

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Agentic Book Recommender")

    with gr.Row():
        # Left Column: Filters
        with gr.Column(scale=1):
            gr.Markdown("### Filters")
            category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
            tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
            
        # Right Column: Chat Interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Llama 3.2 Agent", height=350, type="messages")
            
            with gr.Row():
                user_query = gr.Textbox(label="Ask for a book recommendation:", placeholder="e.g., A sci-fi story about AI...", scale=4)
                submit_button = gr.Button("Send", scale=1)

    gr.Markdown("## Current Recommendations")
    output = gr.Gallery(label="Recommended books", columns=6, rows=3)


    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown, chatbot],
        outputs=[output, chatbot, user_query]
    )
    user_query.submit(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown, chatbot],
        outputs=[output, chatbot, user_query]
    )
    
if __name__ == "__main__":
    dashboard.launch()