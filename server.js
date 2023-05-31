const { spawn } = require('child_process');
const express = require('express');
const path = require('path');

const app = express();
const port = 3000;

//const { renderName } = require('./public/js/script.js')

app.use(express.static(path.join(__dirname, 'public')));

app.post('/py', (req, res) => {
  const script = spawn('python', ['py_scripts/Project2.py']);

  script.stdout.on('data', (data) => {
    console.log(`Script result: ${data}`);
  });

  script.stderr.on('data', (data) => {
    console.error(`Error executing script: ${data}`);
  });

  script.on('close', (code) => {
    console.log(`Script's exit code: ${code}`);
  });
});

app.listen(port, () => {
  console.log("Server listening port 3000");
});