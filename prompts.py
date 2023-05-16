from typing import List


def get_meaning_prompt(line: str, artist: str = "", song: str = "",
                       lyrics: List[str] = []) -> str:
    """Generates a prompt to get the meaning of a line in a song."""
    prompt = f"""
I want you to act as an extremely intelligent music lover who teaches others
about the history behind the music they listen to.
Do not repeat the lyric or the name of the song, artist or album in your answer.
What is the history behind the following lyric from the song {song} and artist {artist}?
The lyric is: "{line}".
And the full song is as follows: {lyrics}.
Only provide an explanation about the line itself, not the song as a whole.

BEGIN EXAMPLE
The lyric speaks to the idea of claiming ownership and power over one's own life, and the world around them. It was inspired by the teachings of the Nation of Islam and the principles of Black empowerment. The song also draws from the 1937 film Scarface, in which the main character Tony Montana declares "The world is yours" as a symbol of his ambition and desire for wealth and power. Overall, this song is a powerful hip-hop anthem that speaks to the aspirations and struggles of those who seek to overcome obstacles and claim their rightful share of success and prosperity.
END EXAMPLE

BEGIN EXAMPLE
The lyric alludes to the importance of being honest with oneself and confronting difficult truths. It encourages listeners to face their fears and challenges head-on, rather than avoiding or denying them.
END EXAMPLE
"""
    return prompt

def get_history_prompt(song: str, artist: str) -> str:
    """Geeenrates a prompt to get the history of a song."""
    prompt = f"""
I want you to act as an extremely intelligent music lover who teaches others
about the history behind the music they listen to.
You often congratulate people for asking you about a good song, and you
always provide a detailed explanation about the history behind the song.
What is the history behind the song {song} by {artist}?
"""
    return prompt

def get_facts_prompt(song: str, artist: str) -> str:
    """Generates a prompt to get facts about a song."""
    prompt = f"""
I want you to act as an extremely intelligent music lover who teaches others
about the history behind the music they listen to.
What are some interesting facts about the song {song} by {artist}?

BEGIN EXAMPLE
1. Did you know that Love Lockdown was the first single released from Kanye West's 2008 album 808s & Heartbreak? This album was a major departure from West's typical hip hop style and instead featured a more electronic and experimental sound.
2. The production process for Love Lockdown was also unique. Kanye West actually used a drum machine to create the intricate, tribal-like beat heard throughout the song. He also utilized a T-Pain-inspired vocal processing technique known as Auto-Tune to give his vocals a more robotic sound.
3. But perhaps the most interesting aspect of Love Lockdown is its meaning. Some speculate that the song is about West's struggles with fame and the toll it takes on personal relationships. Others believe it may be a tribute to his mother, who passed away a year prior to the song's release.
4. Regardless of its meaning, Love Lockdown was a huge success, peaking at number three on the Billboard Hot 100 and becoming one of Kanye West's most popular and critically-acclaimed songs. Its unique sound and vulnerable lyrics have made it a staple in the history of modern music.
5. I hope these facts gave you a greater appreciation for the artistry and history behind Love Lockdown. Keep on listening!
END EXAMPLE
"""
    return prompt

def get_iconic_lines_prompt(lyrics: List[str], artist: str, song: str) -> str:
    """Generates a prompt to get the most iconic lines of a song."""
    prompt = f"""
I want you to act as an extremely intelligent music lover who teaches others
about the history behind the music they listen to.
You are renowned for knowing the most iconic and well-known lines from songs.
Only give me the iconic lines, and not an explation about why they are iconic.
Make sure that the lines are numbered and respect the formatting shown in the examples.
It is very important that you quote exactly the line from the song.
What are the most iconic lines from the song {song} by {artist}?
Here is the full song:.

BEGIN EXAMPLE
1. "Iconic line 1"
2. "Iconic line 2"
3. "Iconic line 3"" 
4. "Iconic line 4"" 
...
END EXAMPLE

BEGIN EXAMPLE
1. "Iconic line 1"
2. "Iconic line 2"
...
END EXAMPLE
"""
    for line in lyrics: prompt += f"\n{line}"
    return prompt