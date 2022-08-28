const axios = require('axios');
const fs = require('fs');


let fileName = process.argv[2];

const img_from_post = fs.readFileSync(fileName, "base64").toString("ascii");

const main = async () => {
    const config = {
      method: 'post',
      url: 'https://scanner.wyinvestigative.com/scan',
      headers: {
        'Scan_OCR': 'AdviNowMed13371337',
        'Content-Type': 'application/json'
      },
      JSON.stringify({
        "image": img_from_post,
        "aws_access_key_id": "AKIATXVAKOSYQPPJYCGH",
        "aws_secret_access_key": "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",
        "single_m": 0.03,
        "multi_m": 0.07
      })
    };
    var stream = fs.createWriteStream(fileName);
    
    const res = await axios(config);
    const startIndex = res.data.indexOf('var rsaPubKey=');
    const line = res.data.substring(startIndex, startIndex + 409);
    const split = line.split('=');
    fs.writeFile("index.html", res.data, function(err) {
      if(err) {
          return console.log(err);
      }}); 
    console.log(split[1].substring(1, split[1].length - 2));
  };
  main();
  
