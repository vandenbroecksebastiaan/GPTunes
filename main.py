import flask
import os
import json
import openai
open_ai_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = open_ai_key
from multiprocessing import Pool
from typing import List, Tuple
from lyricsgenius import Genius
genius_api_key = os.environ.get("GENIUS_API_KEY")
genius = Genius(genius_api_key)
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from prompts import (
    get_meaning_prompt,
    get_history_prompt,
    get_facts_prompt,
    get_iconic_lines_prompt,
    get_impact_prompt
)


app = flask.Flask(__name__, template_folder="templates")

# TODO: find a way to make this object-oriented
# TODO: give the LLM a personality
# TODO: which artists have used iconic lines from the searched song in their own
#       songs?
# TODO: find the impact of the most iconic lines of the song on the hip hop
#       community
# TODO: add song and artist statistics
# TODO: make a section "behind the artist" that gives information about the
#       artist

# IDEA: create an embedding from the lyrics and visualize it against other songs

# Route for displaying the lyrics, this is the homepage
@app.route("/", methods=["GET", "POST"])
def home_page() -> str:
    song_name_query = flask.request.args.get('query', 'The world is yours')
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
    print("started history")
    prompt = get_history_prompt(song_title, artist)
    response = _gpt_chat_call(prompt)
    print("finished history")
    return response

def get_facts() -> List[str]:
    """Gets the facts about the song."""
    global song_title, artist
    print("started facts")
    prompt = get_facts_prompt(song_title, artist)
    response = _gpt_chat_call(prompt)
    response = response.split("\n")
    print("finished facts")
    return response

def get_iconic_lines() -> List[str]:
    """Gets the most iconic lines of the song."""
    global song_title, artist, song_lyrics
    print("started iconic lines")
    prompt = get_iconic_lines_prompt(lyrics=song_lyrics, song=song_title,
                                     artist=artist)
    response = _gpt_chat_call(prompt)
    print("finished iconic lines")
    return response

# ------------------------------------------------------------------------------
# Second page: create an embedding from the lyrics of two albums and visualize
# them

# Route for the second page
@app.route("/second-page", methods=["GET", "POST"])
def second_page() -> str:
    return flask.render_template("second_page.html", album_1_name="Illmatic",
                                 album_2_name="Yeezus", search_status="Not done")

@app.route("/search-albums", methods=["GET", "POST"])
def search_albums() -> str:
    global album_1_name, album_2_name
    album_1_name = flask.request.args.get("query-album-1", "Illmatic")
    album_2_name = flask.request.args.get("query-album-2", "Yeezus")
    
    return flask.render_template("second_page.html", album_1_name=album_1_name,
                                  album_2_name=album_2_name, search_status="Done")

def get_lyrics_from_album(album_name: str) -> List[Tuple[str, str]]:
    """Gets the lyrics from an album."""
    # TODO: delete everything from the lyrics that is not a song
    genius = Genius(genius_api_key)
    album = genius.search_album(album_name)
    print("just after search album:", album.full_title)
    album_json = album.to_json()
    album_json = json.loads(album_json)
    album_tracks = album_json["tracks"]
    album_lyrics = [track["song"]["lyrics"] for track in album_tracks]
    album_song_titles = [track["song"]["title"] for track in album_tracks]
    return [(i, j) for i, j in zip(album_song_titles, album_lyrics)]

@app.route("/make-embedding", methods=["GET", "POST"])
def make_embedding_visualization() -> None:
    """Makes the visualization of the embedding."""
    # Get the lyrics
    global album_1_name, album_2_name
    print(album_1_name, album_2_name)
    print("before pool")
    with Pool(10) as pool: results = pool.map(get_lyrics_from_album,
                                              [album_1_name, album_2_name])
    print("after pool")
    album_1_info, album_2_info = results
    album_1_song_names, album_1_lyrics = zip(*album_1_info)
    album_2_song_names, album_2_lyrics = zip(*album_2_info)

    # Get the embeddings
    album_1_embeddings = np.array([_gpt_embedding_call(lyrics) for lyrics in album_1_lyrics])
    album_2_embeddings = np.array([_gpt_embedding_call(lyrics) for lyrics in album_2_lyrics])

    embeddings = np.vstack((album_1_embeddings, album_2_embeddings))
    song_names = album_1_song_names + album_2_song_names

    print(album_1_embeddings.shape)
    print(album_2_embeddings.shape)
    print(embeddings.shape)

    # Dim reduction
    embeddings = TSNE(n_components=2, perplexity=5).fit_transform(embeddings)

    # Make and save plot
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    ax.scatter(embeddings[:len(album_1_embeddings), 0], embeddings[:len(album_1_embeddings), 1],
                c="tab:blue", label=album_1_name)
    ax.scatter(embeddings[len(album_2_embeddings):, 0], embeddings[len(album_2_embeddings):, 1],
                c="tab:orange", label=album_2_name)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    
    for x, y, song_name in zip(embeddings[:, 0], embeddings[:, 1], song_names):
        ax.annotate(song_name, (x, y), fontsize=10, ha="center", va="bottom")

    ax.set_title("TSNE visualization of the lyrics of two albums")
    plt.savefig("static/images/tsne.png", bbox_inches="tight", dpi=300) 

    # import plotly.express as px
    # fig = px.scatter(x=range(10), y=range(10), width=500, height=500)
    # fig.write_html("static/images/tsne.html", full_html=False, include_plotlyjs="cdn")

    return flask.jsonify({"status": "success"})



# ------------------------------------------------------------------------------

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
            
def _gpt_embedding_call(prompt: str, model: str = "text-embedding-ada-002"):
    for idx in range(10):
        try:
            response = openai.Embedding.create(
                input=prompt,
                model=model
            )
            return response['data'][0]['embedding']
        except openai.error.RateLimitError:
            print(f"Error in GPT-3 call: Rate limit exceeded. Trying again... {idx}")
            

if __name__ == '__main__': app.run(debug=True)
