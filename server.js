const { spawn } = require('child_process');
const express = require('express');
const path = require('path');
const fs = require('fs');
const cheerio = require('cheerio');

const app = express();
const port = 3000;

//const { changeText } = require('./public/js/script.js')

app.use(express.static(path.join(__dirname, 'public')));

app.get('/py', (req, res) => {
  const script = spawn('python', ['py_scripts/Project2.py']);

  script.stdout.on('data', (data) => {
    console.log(`Script result: ${data}`);

    // Render answer
    const html = fs.readFileSync('./public/index.html', 'utf8');
    const $ = cheerio.load(html);

    console.log(` ${data} `);
    $('#result').text(` ${data} `);
    
    const modifiedHtml = $.html();
    
    fs.writeFileSync('./public/index.html', modifiedHtml, 'utf8');
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