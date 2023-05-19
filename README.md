# GPTunes

GPTunes is a (yet) unfinished project (but functional), powered by Flask and GPT-3. Indulge
your senses and broaden your horizons by searching for your favourite songs and
discover lyrics, unravel meanings, explore song histories and be delighted by
fascinating, fun facts.


## Usage
How can you get this working on your computer? You know the deal, put the
Open AI API key in your environment:

```bash
export OPENAI_API_KEY=YOUR_API_KEY_HERE
```

To get the lyrics you should get a token for the [Genius API](https://docs.genius.com/#/authentication-h1)
and do the same thing:

```bash
export GENIUS_API_KEY=YOUR_API_KEY_HERE
```

## Example

There are currently two things that GPTunes can do for you. One the one hand, you can find out more about the lyrics of your favorite songs:

<img width="1278" alt="Screenshot 2023-05-19 at 19 20 36" src="https://github.com/vandenbroecksebastiaan/GPTunes/assets/101555259/6e4bd422-cf7a-4c68-9b67-8b3bcbdefd5e">

On the other hand, you can compare embeddings from two albums. More information is provided on the "album embedding" page. For example, how does a classical Nas album compare to a more modern one such as Magic?

<img width="1278" alt="Screenshot 2023-05-19 at 18 21 02" src="https://github.com/vandenbroecksebastiaan/GPTunes/assets/101555259/58cd1a3d-0083-4757-b7ef-e1e2e6e0da43">
