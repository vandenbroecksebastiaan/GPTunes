function addLyrics() {
    var lyricsElement = document.getElementById("lyrics");
    
    // Get the lines from the server
    axios.post("/get-lyrics")
        .then(function (response) {
            // Add the lines to the lyricsElement
            lines = response.data.lyrics;
            // For every element in lines, add a <p> element to the lyricsElement
            lines.forEach(function(line) {
                var lineElement = document.createElement("p");
                lineElement.setAttribute("id", "lyrics-line");
                lineElement.setAttribute("class", "lyrics-inline");
                lineElement.innerHTML = line;
                lyricsElement.appendChild(lineElement);
            });
        })
        .catch(function (error) {
            console.error('Error:', error);
        });
}

addLyrics();

// Function that gets information about the song
function getSongInfo() {
    axios.post('/song-info')
        .then(function (response) {
            updateHistory(response);
            updateFacts(response);
            updateIconicLines(response);
        })
}

function updateHistory(response) {
    var historyElement = document.getElementById('history');
    historyElement.innerHTML = response.data.history;
}

function updateIconicLines(response) {
    var iconicLinesElement = document.getElementById('iconic-lines');
    iconicLinesElement.innerHTML = response.data.iconic_lines.join('<br>');
    
    // Iterate over #lyrics and make the line red if it is an iconic line
    var lines = document.querySelectorAll('#lyrics');
    var iconic_lines = response.data.iconic_lines;

    var iconic_lines = iconic_lines.map(function(element) {
        return element.substring(3)
                      .replace('"', '').replace('"', '')
                      .toLowerCase();
    });

    // Select lyrics by id
    var lyrics = document.getElementById("lyrics");
    // Iterate over children of lyrics
    var lines = lyrics.children;
    var lines = Array.from(lines);

    lines.forEach(function(line) {
        if (iconic_lines.includes(line.textContent.toLowerCase())) {
            line.classList.add('iconic');
        }
    });
}

function updateFacts(response) {
    var factsElement = document.getElementById('facts');
    // Add a linebreak after each element of factsElement
    factsElement.innerHTML = response.data.facts.join('<br>');
}

getSongInfo();

// Function to handle line clicked
function lineClicked(event) {
    // Remove the 'selected' class from all lines
    var lines = document.querySelectorAll('#lyrics-line');
    lines.forEach(function(line) {
        line.classList.remove('selected');
    });
    
    // Get the clicked line element
    var clickedLine = event.target;
    
    // Add the 'selected' class to the clicked line
    clickedLine.classList.add('selected');
    
    // Get the text content of the clicked line
    var lineText = clickedLine.textContent;

    // Call the lineClicked function
    axios.post('/line-clicked', { line: lineText })
        .then(function (response) {
            updateMeaning(response);
        })
        .catch(function (error) {
            console.error('Error:', error);
        });
}

function updateMeaning(response) {
    var meaningElement = document.getElementById('meaning');
    meaningElement.innerHTML = response.data.meaning;
}

// Get all the lines in the lyrics
var lines = document.querySelectorAll('#lyrics');

// Add click event listeners to each line
lines.forEach(function(line) {
    line.addEventListener('click', lineClicked);
});