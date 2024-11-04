def has_gendered_substr(text) -> bool:
    male_substr = ["man", "guy", "men", "boys", "my wife"]
    female_substr = ["woman", "girl", "women", "girls", "my husband", "feminine", "female"]
    for substr in male_substr:
        for word in text.lower().split(' '):
            if substr == word.strip(".,!?"):
                return True
    for substr in female_substr:
        for word in text.lower().split(' '):
            if substr == word.strip(".,!?"):
                return True
    return False

def has_age_substr(text) -> bool:
    raise NotImplementedError("Not implemented")

def has_socio_substr(text) -> bool:
    raise NotImplementedError("Not implemented")

def has_direct_substr(text) -> bool:
    return has_gendered_substr(text) or has_age_substr(text) or has_socio_substr(text)