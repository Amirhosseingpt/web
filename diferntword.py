def remove(word):
    word = word.replace("'" , "").replace('"', "").replace('"', "").replace("'", "").replace(" ", "").replace("?" , "")

    return word

text = "awdakwnawklnf?????'''"
print(remove(text))