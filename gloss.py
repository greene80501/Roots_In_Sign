def to_gloss(text, names=None):
    if names is None:
        names = []
    # Step 1: Normalize and split the text
    words = text.strip().split()
    glossed = []

    for word in words:
        clean_word = word.strip(".,?!").lower()
        if clean_word in [n.lower() for n in names]:
            # Fingerspell the name
            fingerspelled = ' '.join(list(clean_word.upper()))
            glossed.append(fingerspelled)
        else:
            glossed.append(clean_word.upper())

    return ' '.join(glossed)


if __name__ == "__main__":
    _sentence = "Yesterday Wyatt went to the store."
    _names = ["Wyatt"]
    print(to_gloss(_sentence, _names))
