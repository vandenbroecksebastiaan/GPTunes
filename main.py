import flask
import os
import openai
open_ai_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = open_ai_key
from multiprocessing import Pool
from typing import List
from lyricsgenius import Genius
genius_api_key = os.environ.get("GENIUS_API_KEY")
genius = Genius(genius_api_key)

from prompts import (
    get_meaning_prompt,
    get_history_prompt,
    get_facts_prompt,
    get_iconic_lines_prompt,
    get_impact_prompt
)


app = flask.Flask(__name__)

# TODO: find a way to make this object-oriented
# TODO: give the LLM a personality
# TODO: which artists have used iconic lines from the searched song in their own
#       songs?
# TODO: find the impact of the most iconic lines of the song on the hip hop
#       community
# TODO: add song and artist statistics
# TODO: make a section "behind the artist" that gives information about the
#       artist

# Route for displaying the lyrics
@app.route("/", methods=["GET", "POST"])
def lyrics() -> str:
    song_name_query = flask.request.args.get('query', 'The world is yours')
    song = genius.search_song(song_name_query)
    
    global song_title, artist, song_lyrics
    song_lyrics = song.lyrics
    song_title = song.title
    album_art_url = song.header_image_url
    artist = song.primary_artist.name
    song_lyrics = song_lyrics.split("\n")[1:]
    
    print(f"song name: {song_title}, artist: {artist}")

    # Pass the lyrics to the HTML template
    return flask.render_template('lyrics.html', lines=song_lyrics,
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
    response = _gpt_call(prompt)
    return response

def get_history() -> str:
    """Gets the history behind the song."""
    global song_title, artist
    print("started history")
    prompt = get_history_prompt(song_title, artist)
    response = _gpt_call(prompt)
    print("finished history")
    return response

def get_facts() -> List[str]:
    """Gets the facts about the song."""
    global song_title, artist
    print("started facts")
    prompt = get_facts_prompt(song_title, artist)
    response = _gpt_call(prompt)
    response = response.split("\n")
    print("finished facts")
    return response

def get_iconic_lines() -> List[str]:
    """Gets the most iconic lines of the song."""
    global song_title, artist, song_lyrics
    print("started iconic lines")
    prompt = get_iconic_lines_prompt(lyrics=song_lyrics, song=song_title, artist=artist)
    response = _gpt_call(prompt)
    print("finished iconic lines")
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
