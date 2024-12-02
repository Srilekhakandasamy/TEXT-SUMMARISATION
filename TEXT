# TEXT-SUMMARISATION
# Install the required libraries
!pip install transformers gradio -q

# Import necessary libraries
from transformers import pipeline
import gradio as gr

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define the summarization function
def summarize_text(input_text, min_length=25, max_length=150):
    if len(input_text.strip()) == 0:
        return "Please provide some text to summarize."
    summary = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

# Create the Gradio interface
interface = gr.Interface(
    fn=summarize_text,
    inputs=[
        gr.Textbox(lines=10, label="Input Text", placeholder="Paste your text here..."),
        gr.Slider(10, 50, value=25, label="Minimum Summary Length"),
        gr.Slider(50, 300, value=150, label="Maximum Summary Length"),
    ],
    outputs=gr.Textbox(label="Summarized Text"),
    title="Text Summarization App",
    description="Enter text in the box and get a summarized version of it. Adjust the sliders to change the summary length.",
)

# Launch the app
interface.launch()
