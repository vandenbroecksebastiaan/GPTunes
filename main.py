import flask
import os
import json
import openai
open_ai_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = open_ai_key
from multiprocessing import Pool
from typing import List, Tuple
import numpy as np
import umap
import matplotlib.pyplot as plt

from prompts import (
    get_meaning_prompt,
    get_history_prompt,
    get_facts_prompt,
    get_iconic_lines_prompt
)

genius_api_key = os.environ.get("GENIUS_API_KEY")

app = flask.Flask(__name__, template_folder="templates")

# TODO: which artists have used iconic lines from the searched song in their own
#       songs?
# https://developer.musixmatch.com/documentation/api-reference/matcher-track-get
# TODO: add song and artist statistics
# TODO: make a section "behind the artist" that gives information about the
#       artist
# TODO: implement a crawl through a graph structure of some sorts and
#       compare embeddings to find similar songs

@app.route("/", methods=["GET", "POST"])
def home_page() -> str:
    song_name_query = flask.request.args.get('query', 'The world is yours')
    from lyricsgenius import Genius
    access_token = os.environ.get("GENIUS_API_KEY")
    genius = Genius(access_token, timeout=10, sleep_time=0.1, verbose=True,
                    retries=5)
    song = genius.search_song(song_name_query)
    
    global song_title, artist, song_lyrics
    song_lyrics = song.lyrics
    song_title = song.title
    album_art_url = song.header_image_url
    artist = song.primary_artist.name
    song_lyrics = song_lyrics.split("\n")[1:]

    # Pass the lyrics to the HTML template
    return flask.render_template('home_page.html', lines=song_lyrics,
                                 search_query=song_name_query,
                                 album_art_url=album_art_url)

@app.route("/get-lyrics", methods=["GET", "POST"])
def get_lyrics() -> flask.Response:
    print("get_lyrics called")
    global song_lyrics
    return flask.jsonify({"status": "success", "lyrics": song_lyrics})

# Route for handling the line clicked information
@app.route("/line-clicked", methods=["POST"])
def line_clicked() -> flask.Response:
    print("Getting the meaning behind the clicked line B)")
    request_data = flask.request.get_json()
    clicked_line = request_data["line"]
    meaning = get_meaning(clicked_line)
    return flask.jsonify({"status": "success", "meaning": meaning})
    
@app.route("/song-info", methods=["GET", "POST"])
def song_searched() -> flask.Response:
    print("getting the history, facts and iconic lines >:)")

    pool = Pool(10)
    history = pool.apply_async(get_history)
    facts = pool.apply_async(get_facts)
    iconic_lines = pool.apply_async(get_iconic_lines)
    history = history.get(); facts = facts.get(); iconic_lines = iconic_lines.get();
    pool.close(); pool.join();
    
    iconic_lines = iconic_lines.split("\n")
    return flask.jsonify({"status": "success", "history": history, "facts": facts,
                          "iconic_lines": iconic_lines})

def get_meaning(clicked_line="") -> str:
    """Gets the history behind the clicked line."""
    global song_title, artist, song_lyrics
    prompt = get_meaning_prompt(clicked_line, artist, song_title, song_lyrics)
    response = _gpt_chat_call(prompt)
    return response

def get_history() -> str:
    """Gets the history behind the song."""
    global song_title, artist
    prompt = get_history_prompt(song_title, artist)
    response = _gpt_chat_call(prompt)
    return response

def get_facts() -> List[str]:
    """Gets the facts about the song."""
    global song_title, artist
    prompt = get_facts_prompt(song_title, artist)
    response = _gpt_chat_call(prompt)
    response = response.split("\n")
    return response

def get_iconic_lines() -> List[str]:
    """Gets the most iconic lines of the song."""
    global song_title, artist, song_lyrics
    prompt = get_iconic_lines_prompt(lyrics=song_lyrics, song=song_title,
                                     artist=artist)
    response = _gpt_chat_call(prompt)
    return response

@app.route("/album-embedding", methods=["GET", "POST"])
def second_page() -> str:
    return flask.render_template("album_embedding.html", album_1_name="Illmatic",
                                 album_2_name="Yeezus", search_status="Not done")

@app.route("/search-albums", methods=["GET", "POST"])
def search_albums() -> str:
    global album_1_name, album_2_name
    album_1_name = flask.request.args.get("query-album-1", "Illmatic")
    album_2_name = flask.request.args.get("query-album-2", "Yeezus")
    
    return flask.render_template("album_embedding.html", album_1_name=album_1_name,
                                  album_2_name=album_2_name, search_status="Done")

@app.route("/make-lyric-embedding", methods=["GET", "POST"])
def make_lyric_embedding_visualization() -> None:
    """Makes the visualization of the embedding."""
    from lyrics2vec import LyricsEmbedder
    global album_1_name, album_2_name
    embedder = LyricsEmbedder([album_1_name, album_2_name])
    embedder.get_lyrics()
    embedder.get_embeddings()
    embedder.make_visualization()

    return flask.jsonify({"status": "success"})

@app.route("/make-music-embedding", methods=["GET", "POST"])
def make_music_embedding_visualization() -> None:
    from music2vec import MusicEmbedder
    global album_1_name, album_2_name

    embedder = MusicEmbedder([album_1_name, album_2_name])
    # Sometimes there are no previews available for an album
    preview_status = embedder.download_songs()
    print(preview_status)
    if preview_status == "error": return flask.jsonify({"status": "error"})
    embedder.downsample_songs()
    embedder.to_embedding()
    embedder.reduce_embeddings()
    embedder.make_visualization()

    return flask.jsonify({"status": "success"})

def _gpt_chat_call(prompt: str, max_tokens: int = 500, model: str = "gpt-3.5-turbo"):
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
            

if __name__ == '__main__':
    app.run(debug=True)
