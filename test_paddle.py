import easyocr

reader = easyocr.Reader(['hi'], gpu=True)  # or ['hi', 'ne'] if you want to try Nepali too

result = reader.readtext('test.png')

# Extract only the text parts
texts = [detection[1] for detection in result]

print(texts)
# Output: ['१९१८२५ ३१', 'अनिता तामाङ']