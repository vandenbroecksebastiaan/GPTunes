import requests
import wget
from tqdm import tqdm
import os
from transformers import Wav2Vec2Processor, Data2VecAudioModel
import torch
from datasets import Dataset, Audio
import umap
import matplotlib.pyplot as plt
import numpy as np


class MusicEmbedder:
    def __init__(self, album_names):
        self.album_names = album_names
        self.song_data = [] # [[album_name, song_name, song_path, embedding]]
    
    def download_songs(self):
        """Gets the preview of the song."""
        url = "https://accounts.spotify.com/api/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": "2b8eed681f1746e290fef8574c11d303",
            "client_secret": "ab4b4d88e5da41d29b3d987a7fa98a10"
        }
        response = requests.post(url, data=data)
        token_data = response.json()
        access_token = token_data["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}
        
        for album_name in self.album_names:
            url = f"https://api.spotify.com/v1/search?q={album_name}" \
                   "&type=album&limit=1&include_external=audio"
            response = requests.get(url, headers=headers).json()["albums"]
            print(response["items"][0]["name"])
            print(response["items"][0]["artists"])
            
            album_id = response["items"][0]["id"]
            url = f"https://api.spotify.com/v1/albums/{album_id}/tracks?limit=50"
            response = requests.get(url, headers=headers).json()["items"]
            
            # If there are no previews, we return an error
            if response[0]["preview_url"] is None: return "error"
            
            track_names = [i["name"] for i in response]
            track_preview_urls = [i["preview_url"] for i in response]
            
            os.makedirs(f"previews/{album_name}", exist_ok=True)
            for file in os.listdir(f"previews/{album_name}"):
                os.remove(f"previews/{album_name}/{file}")
            
            for url, song_name in tqdm(zip(track_preview_urls, track_names),
                                       desc="Downloading previews"):
                path = f"previews/{album_name}/{song_name}.mp3"
                self.song_data.append([album_name, song_name, path])
                wget.download(url, path)
    
    def to_embedding(self):
        processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
        model = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1").cuda()
        
        pbar = tqdm(enumerate(self.song_data), desc="Generating embeddings",
                    total=len(self.song_data))
        for idx, (album_name, song_name, path) in pbar:
            dataset = Dataset.from_dict({"audio": [path]})\
                             .cast_column("audio", Audio())
            inputs = processor(dataset[0]["audio"]["array"], sampling_rate=16000,
                               return_tensors="pt")
            inputs["input_values"] = inputs["input_values"].cuda()
            inputs["attention_mask"] = inputs["attention_mask"].cuda()

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
            time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
            time_reduced_hidden_states = time_reduced_hidden_states.cpu().flatten().tolist()
            
            # Add the embedding to the song data
            self.song_data[idx] = [album_name, song_name, path,
                                   time_reduced_hidden_states]

    def reduce_embeddings(self):
        embeddings = np.array([i[3] for i in self.song_data])
        red_embeddings = umap.UMAP(n_neighbors=5).fit_transform(embeddings)
        red_embeddings = (red_embeddings - red_embeddings.mean(axis=0))\
                         / red_embeddings.std(axis=0)
        for idx, embedding in enumerate(red_embeddings):
            self.song_data[idx][3] = embedding.tolist()
        print(self.song_data[:5])
    
    def make_visualization(self):
        """Creates a visualization of the embeddings.""" 
        # Create a color map
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                  "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
        album_to_color = {}
        for idx, album_name in enumerate(self.album_names):
            album_to_color[album_name] = colors[idx]
            
        # Prevent parse error
        for idx in range(len(self.song_data)):
            self.song_data[idx][1] = self.song_data[idx][1].replace("$", "S")

        fig, ax = plt.subplots(figsize=(7, 7))
        for album_name, song_name, path, embedding in self.song_data:
            x, y = embedding
            ax.scatter(x, y, c=album_to_color[album_name])
            ax.annotate(song_name, (x, y), fontsize=8, ha="center", va="bottom",
                        c="white")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title("")

        plt.savefig("static/images/music_embedding.png", bbox_inches="tight",
                    dpi=300, transparent=True)


if __name__ == "__main__":
    pass
