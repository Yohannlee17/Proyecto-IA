document.getElementById('chat-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    var input = document.getElementById('chat-input');
    var message = input.value.trim();
    if (message) {
        appendMessage('user-message', message);
        input.value = '';
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            appendMessage('bot-message', data.response);
        })
        .catch(error => console.error('Error:', error));
    }
}

function appendMessage(className, message) {
    var chatBody = document.getElementById('chat-body');
    var messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);
    messageDiv.textContent = message;
    chatBody.appendChild(messageDiv);
    chatBody.scrollTop = chatBody.scrollHeight; // Scroll to the bottom
}
