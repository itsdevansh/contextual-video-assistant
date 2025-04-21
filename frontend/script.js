document.addEventListener('DOMContentLoaded', () => {
    const videoPlayer = document.getElementById('videoPlayer');
    const queryInput = document.getElementById('queryInput');
    const submitButton = document.getElementById('submitQuery');
    const answerDisplay = document.getElementById('answerDisplay');
    const timestampSuggestion = document.getElementById('timestampSuggestion');
    const loadingIndicator = document.getElementById('loadingIndicator');

    // --- Configuration ---
    // Update this if your backend runs on a different port or host
    const BACKEND_URL = 'http://localhost:5000/query';

    submitButton.addEventListener('click', handleQuerySubmit);
    queryInput.addEventListener('keypress', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleQuerySubmit();
        }
    });


    async function handleQuerySubmit() {
        const query = queryInput.value.trim();
        if (!query) {
            alert('Please enter a question.');
            return;
        }

        // Clear previous results and show loading
        answerDisplay.textContent = '';
        timestampSuggestion.innerHTML = ''; // Clear previous buttons/links
        loadingIndicator.style.display = 'block';
        submitButton.disabled = true;

        try {
            const response = await fetch(BACKEND_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            });

            if (!response.ok) {
                // Try to parse error message from backend if available
                let errorMsg = `Error: ${response.status} ${response.statusText}`;
                try {
                     const errorData = await response.json();
                     errorMsg = `Error: ${errorData.error || errorMsg}`;
                } catch (e) { /* Ignore parsing error */ }
                throw new Error(errorMsg);
            }

            const data = await response.json();

            // Display the LLM answer
            answerDisplay.textContent = data.answer || 'No answer provided.';

            // Display the timestamp suggestion if available
            if (data.suggested_timestamp !== null && data.suggested_timestamp !== undefined) {
                displayTimestampSuggestion(data.suggested_timestamp);
            } else {
                 timestampSuggestion.textContent = 'No specific timestamp suggestion available.';
            }

        } catch (error) {
            console.error('Error fetching data:', error);
            answerDisplay.textContent = `Failed to get answer: ${error.message}`;
            timestampSuggestion.textContent = ''; // Clear timestamp on error
        } finally {
            // Hide loading and re-enable button
            loadingIndicator.style.display = 'none';
            submitButton.disabled = false;
        }
    }

    function displayTimestampSuggestion(timeInSeconds) {
        // Clear previous content
        timestampSuggestion.innerHTML = '';

        const timeButton = document.createElement('button');
        const minutes = Math.floor(timeInSeconds / 60);
        const seconds = Math.floor(timeInSeconds % 60);
        timeButton.textContent = `Jump to relevant point (<span class="math-inline">\{minutes\}\:</span>{seconds.toString().padStart(2, '0')})`;

        timeButton.addEventListener('click', () => {
            if (videoPlayer) {
                videoPlayer.currentTime = timeInSeconds;
                videoPlayer.play(); // Optional: start playing automatically
            }
        });

        timestampSuggestion.appendChild(timeButton);
    }

});