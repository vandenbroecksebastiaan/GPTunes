from flask import Flask, render_template, request, redirect, url_for, jsonify

import os
import openai
open_ai_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = open_ai_key
from multiprocessing import Pool
from typing import List

from lyricsgenius import Genius
genius_api_key = os.environ.get("GENIUS_API_KEY")
genius = Genius(genius_api_key)

from prompts import get_meaning_prompt, get_history_prompt, get_facts_prompt 

app = Flask(__name__)

# TODO: put the most iconic lines of the song in a different color
# TODO: give the LLM a personality

# Route for displaying the lyrics
@app.route("/", methods=["GET", "POST"])
def lyrics():
    song_name_query = request.args.get('query', 'The world is yours')
    song = genius.search_song(song_name_query)
    global song_title, artist, song_lyrics
    song_lyrics = song.lyrics
    song_title = song.title
    artist = song.primary_artist.name
    song_lyrics = song_lyrics.split("\n")[1:]

    # Pass the lyrics to the HTML template
    return render_template('lyrics.html', lines=song_lyrics, search_query=song_name_query)

@app.route('/')
def search():
    search_query = request.args.get('query', '')
    print('Search query:', search_query)
    return redirect(url_for('', query=search_query))

# Route for handling the line clicked information
@app.route('/line-clicked', methods=['POST'])
def line_clicked():
    request_data = request.get_json()
    clicked_line = request_data['line']
    
    pool = Pool()
    meaning = pool.apply_async(get_meaning, [clicked_line])
    history = pool.apply_async(get_history)
    facts = pool.apply_async(get_facts)
    meaning = meaning.get(); history = history.get(); facts = facts.get();
    pool.close(); pool.join();
    
    return jsonify({"status": "success", "meaning": meaning, "history": history,
                    "facts": facts})

def get_meaning(clicked_line="") -> str:
    """Gets the history behind the clicked line."""
    global song_title, artist, song_lyrics
    prompt = get_meaning_prompt(clicked_line, artist, song_title, song_lyrics)
    response = _gpt_call(prompt)
    return response

def get_history() -> str:
    """Gets the history behind the song."""
    global song_title, artist
    prompt = get_history_prompt(song_title, artist)
    response = _gpt_call(prompt)
    return response

def get_facts() -> List[str]:
    """Gets the facts about the song."""
    global song_title, artist
    prompt = get_facts_prompt(song_title, artist)
    response = _gpt_call(prompt)
    response = response.split("\n")
    return response

def _gpt_call(prompt: str, max_tokens: int = 500, model: str = "gpt-3.5-turbo"):
    for idx in range(10):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response["choices"][0]["message"]["content"]
        except openai.error.RateLimitError:
            print(f"Error in GPT-3 call: Rate limit exceeded. Trying again... {idx}")

if __name__ == '__main__': app.run(debug=True)
