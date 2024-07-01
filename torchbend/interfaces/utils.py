import random

def get_random_hash(n=8):
    return "".join([chr(random.randrange(97,122)) for i in range(n)])
