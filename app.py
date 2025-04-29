from flask import Flask, request, render_template,jsonify,session
import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from utlis import *
from dotenv import load_dotenv
import google.generativeai as genai


# Load the environment variables from the .env file
load_dotenv()
# Get the API key from environment variables
api_key = os.getenv('GOOGLE_API_KEY')
flask_secret_key = os.getenv('FLASK_SECRET_KEY')

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')
     


app = Flask(__name__)
app.secret_key = flask_secret_key  # Set the secret key for session management

# Load the vocabulary from the file
with open("vocab.json", "r") as f:
    vocab = json.load(f)

# Initialize the TextVectorization layer with the loaded vocabulary
vectorization = TextVectorization(
    max_tokens=len(vocab),
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)
vectorization.set_vocabulary(vocab)

# Continue with the rest of your inference code
loaded_model = tf.saved_model.load('image_captioning_model1')
index_lookup = dict(zip(range(len(vocab)), vocab))

def generate_caption(sample_img_path):
    # Check if the image file exists
    if not os.path.isfile(sample_img_path):
        print(f"Error: The file {sample_img_path} does not exist.")
        return

    # Read the image from the disk
    sample_img = decode_and_resize(sample_img_path)

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = loaded_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = loaded_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start>"
    for i in range(SEQ_LENGTH):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        tokenized_caption = tf.cast(tokenized_caption, dtype=tf.float32)  # Ensure dtype is tf.float32
        mask = tf.cast(tf.math.not_equal(tokenized_caption, 0), dtype=tf.bool)  # Ensure mask is boolean

        predictions = loaded_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )

        # Get the predicted token index
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup.get(sampled_token_index, "<unknown>")
        
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    return decoded_caption


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename	
        img.save(img_path)
        description=generate_caption(img_path)
        session['description'] = description  # Store the description in the session

    return render_template("index.html", prediction = description, img_path = img_path)

def get_response(prompt):
    response = model.generate_content(prompt)
    return response.text

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    topic = data.get('topic')
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    
    prompt = f"Compose a captivating story about {topic} in English. Your story should be engaging and easy to read, with vivid descriptions and compelling characters. Aim for a length of approximately 200 words."
    response_text = get_response(prompt)
    return jsonify({"response": response_text})



if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)    
     

