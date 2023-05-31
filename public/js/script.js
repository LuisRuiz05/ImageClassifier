const button = document.getElementById('call_btn');
const text = document.getElementById('result');

button.addEventListener('click', function() {
  text.textContent = 'Text Changed!';
  fetch('/py', { method: 'POST' })
    .then(response => {
        console.log('Correcto');
    })
    .catch(error => {
        console.error('Error', error);
    });
});