#Install necessary libraries
!pip install transformers torch sentence-splitter
import nltk
nltk.download('punkt')
from transformers import pipeline

summarizer = pipeline("summarization", model = "facebook/bart-large-cnn")

def summarize_passage(text, num_bullets = 5):
  """
  Generate bullet summary of provided input text using pre-trained model.

  Parameters:
  text: (str) the passage you want summarized
  num_bullets: (int) the number of bulllets you want for the summary

  return:

  (str) formatted string with bullet points of summarization
  """

  prompt = ( f"summarize this passage into exactly {num_bullets} bullet points, with key points. No introductory title sentences." f"\n\nPASSAGE:\n{text}")
  summary = summarizer(
    prompt,
    max_length = 150,
    min_length = 60,
    do_sample = False, #uses deterministic beam search(more reliable)
    truncation = True
  )[0]['summary_text']
  #summarize this passage into exactly {num_bullets} bullet points, with key points. No introductory title sentences.
  bullet_points = [f"*{line.strip()}" for line in summary.split('\n') if line.strip()]
  return "\n".join(bullet_points)
