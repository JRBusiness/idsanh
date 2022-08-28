from os import name
import re

from cv2 import add, validateDisparity

result_dict = {
    "Address": "2. 22 asdqwefe",
    "Name": ". fewfw wfwfw",
    "fname": " fwfw wdwdw",
    "Address2": "2 232 fefefe",
    "Address3": "343 efefe efe",
    "Address4": "23b wfef egege",
    "namama": "fefwwf fwfwfw",
    "dasda": "DD3ruf3huff3fi3",
    "wddwdjw": "933/22/2222",
    "wddwdsjw": "9. 33/22/2222",
    "dwdwdw": "232. 33-22-2222"
}
addy_string = re.compile(r'^(\d+\W?|\W?)\s\d')
name_string = re.compile(r'^(\W|\d+\w?|\s?)\s?\w')
date_string = re.compile(r'.*?(\d{2}\/?\-?\d{2}\/?\-?\d{4})')
for key, value in result_dict.items():
    if key == "Address" or key == "Address2" or key == "Address3" or key == "Address4":
        if addy_string.match(value):
            result_dict[key] = value.strip(addy_string.search(value).group(1)).strip(r'^ ')
    else:
        if name_string.match(value):
            if date_string.match(value):
                result_dict[key] = re.search(date_string, value).group(1)
            else:
                result_dict[key] = value.strip(name_string.search(value).group(1)).strip(r'^ ')
    print(result_dict[key])
