 import random
from datetime import datetime, timedelta
import time
import ast
import torch
import operator
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

#Summarizer setup
#summarizer= pipeline("summarization", model = "facebook/bart-large-cnn")
model_name = "facebook/bart-large-cnn"
device = "cuda" if torch.cuda.is_avalible() else "cpu"

try:
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  print("Loaded successfully")
except Exception as e:
  print(f"Error {e}")

def summarize_passage(text, num_bullets = 5):
  inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(device)
  #not using pipeline
  summary_ids = model.generate(
      inputs["input_ids"],
      num_beams=4,
      max_length=150,
      min_length=60,
      early_stopping=True
  )
  summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  #summarize this passage into exactly {num_bullets} bullet points, with key points. No introductory title sentences.
  #TODO
  bullet_points = [f"*{line.strip()}" for line in summary.split('\n') if line.strip()]
  return "\n".join(bullet_points)

VALID_OPERATORS ={
    ast.Add:operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow
}
def eval_node(node):
  """Safely process Ast nodes"""
  if isinstance(node, ast.Constant):
    return node.value
  elif isinstance(node, ast.BinOp): #checks for math signs
    return VALID_OPERATORS[type(node.op)](eval_node(node.left), eval_node(node.right))
  elif isinstance(node, ast.UnaryOp): #checks if uni like -5
    return VALID_OPERATORS[type(node.op)](eval_node(node.operand))
  raise TypeError(node)

def sentence_to_words(sentence):
    #Convert a sentence into  lowercase words
    return set(sentence.lower().split())


def similarity(q1, q2):
    # similarity count of matching words
    words1 = sentence_to_words(q1)
    words2 = sentence_to_words(q2)
    return len(words1.intersection(words2))
def is_math_expression(text):
  """
  Check if the user input looks like a math expression.
  Allowed characters: digits, + - * / ( ) . and spaces.
  """
  allowed_chars =set("1234567890+-*/().**")
  text_stripped = text.replace(" ", "")
  if not text_stripped:
    return False
  if not any(ch.isdigit() for ch in text_stripped):
    return False
  return all(ch in allowed_chars for ch in text_stripped)
def evaluate_math(text):
  try:
    node = ast.parse(text, mode="eval")
    result= eval_node(node.body)
    return f"The answer is {result}"
  except Exception:
    return """Uhh, that doesn't look like a valid math expression. Either I'm wrong, or you had a typo...
    Or maybe you should lower your expectations."""




# Training data with multiple questions for the same answer
training_data = {
    "Hello!": [
        "hi",
        "hello",
        "hey",
        "yo",
        "hi there",
        "wassup",
        "hi gang"
    ],

    "I am doing well. Thanks for asking.": [
        "how are you",
        "how are you doing",
        "are you ok",
        "are you okay",
        "how you doing",
        "how ya feeling",
        "ya feelin fresh",
        "you feeling fresh"
        "you feeling good"

    ],

    "Python is a programming language used for many types of projects.": [
        "what is python",
        "explain python",
        "tell me about python"
    ],
    "INFO": [
        "what is your name",
        "who are you",
        "tell me your name",
        "info"
    ],


    "Goodbye.": [
        "bye",
        "goodbye",
        "see you",
        "later",
        "bye bye"
    ],
    "JOKE_REQUEST":
     [
         "tell me a joke",
         "tell me something funny",
         "i want to laugh",
         "let's joke!",
         "joke"
     ],

}

jokes = ["How do you know if a vampire is unwell? Because he'll be coffin.", "Where do pirates get their hooks? Second hand shops.", "Why did the bicycle collapse? It was too tired.", "What kind of music do bubbles hate? Pop.", "Why did the hairdresser win the race? He knew a shortcut.", "Why did the bullet lose its job? It got fired."]
info= ["I'm ChadGPT, the greatest ai chatbot the world has ever seen!", "I'm ChadGPT, the most efficient and cool chatbot ever! (RAM not included)", "I'm ChadGPT, ChatGPT if it was good!"]
def get_best_answer(user_input):
    best_score = -1
    best_answer = "I am not sure I understand. Please try asking in another way."

    for answer, question_list in training_data.items():
        for question in question_list:
            score = similarity(user_input, question)

            if score > best_score:
                best_score = score
                best_answer = answer
    if best_answer == "JOKE_REQUEST":
      return random.choice(jokes)
    if best_answer == "INFO":
      return random.choice(info)
    else:
      return best_answer if best_score > 0 else "I am not sure I understand. Please try again."


print("ChadGPT Ready for epic tech action.")
print("Type 'quit' to stop.\n")

while True:
    user = input("You: ")
    user = user.lower().strip()
    if user == "quit":
      print("Bot: Goodbye.")
      break
    elif user.startswith("summarize:"):
      passage = user[10:].strip()
      if len(passage)<20:
        print("Bot: The passage is too short to summarize! Find something better!")
      else:
        print("Bot: Let's see this passage...")
        print(summarize_passage(passage))
    elif user in ["time", "what time is it", "tell me the time", "current time", "what is the time"]:
      now = datetime.now() -timedelta(hours=5)
      print(now.strftime("The time is %I:%M:%S %p"))
    elif user in ["day", "what day is it", "tell me the date", "current date", "what is the date", "date"]:
      date = datetime.now()
      print(date.strftime("The date is %Y-%m-%d."))
    elif is_math_expression(user):

      print("Bot:", evaluate_math(user))
    else:
      print("Bot:", get_best_answer(user)) 
