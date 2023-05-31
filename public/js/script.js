const button = document.getElementById('call_btn');
const text = document.getElementById('result');

button.addEventListener('click', function() {
    fetch('/py', { method: 'GET' })
});