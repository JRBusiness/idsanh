# id-scanner

  

This service provides id and insurance card scanning with optical character recognition, barcode scanning, and properly parsing the info.

For example, given an image of a drivers license, it responds with a json object containing its information.

  
  

## SETUP INSTRUCTIONS:

create `.env` file at the root of the repository, please refer to `server/settings.py`
 to see what info the `.env` file should contains.
  

### Setup virtual environment

---
```bash

$ pipenv shell

$ pipenv install -r requirements/local.txt

```

### Running the API server

To run with uvicorn, to use ssl, you can pass in the key and cert file.

    uvicorn server.api.scanner.views:app --ssl-keyfile=./key.pem --ssl-certfile=./cert.pem

To run with gunicorn, you can use the certificate by passing the key and cert file.
```

$ gunicorn --keyfile=./key.pem --certfile=./cert.pem -k uvicorn.workers.UvicornWorker server.api.scanner.views:app

```

### Client Usage
The id scanner serving 3 functions through the following end point:

> /scan_id
> /scan_insurance
> /barcode

We authorize each request using our custom JWT auth. The api key must be passed in from header `Authorization` . We could use username and password for authorization as well. Please refer to `server/shared/auth/auth.py` for more authorization options.
header

    -H 'Authorization: SomeKeyIn.env_file'

The payload to be send to our endpoint include only the image data in `bytes` format. To create the proper image data, we need to first open and read image, then decode it with `base64`, and finally encode with `ASCII`. Below are the samples on how to encode the proper image data in Python and JavaScript.
    Python:

    with open(file, "rb") as imgage:  
	    img = base64.b64encode(imgage.read())
	    image_payload = img.decode("ascii")

JavaScript

    image_payload = fs.readFileSync(fileName, "base64").toString("ascii");

### The Response
The response will be parsed properly in Json format.

    {  
      "error": "None",  
      "success": "None",  
      "parsedResult": {  
	    "height": "5-8",  
        "exp": "04-27-2011",  
        "first_name": "",  
        "address": "",  
        "city_state_zip": "",  
        "last_name": "LASTNAME",  
        "id_type": "LICENSE",  
        "restrictions": "8",  
        "eyes": "BR",  
        "sex": "F",  
        "class": "D",  
        "iss": "04-27-2010",  
        "no.": "99999999",  
        "state_name": "",  
        "dob": "04-27-1970"  
      }  
    }
