import re

us_spelling = set([
    "color", "honor", "favorite", "theater", "center", "meter", 
    "defense", "license", "analyze", "realize", "organize", 
    "traveler", "jewelry", "program", "aluminum", "plow", "catalog", 
    "gray", "leukemia", "tire",  "fall", "thanksgiving", "memorial day", 
    "labor day", "fourth of july", "veterans day", "super bowl", 
    "nba finals", "world series", "washington dc", "new york", "california",
    "chicago", "state", "governor", "semester","elementary school", 
    "high school", "college", "fiscal year"
])

british_spelling = set([
    "colour", "honour", "favourite", "theatre", "centre", "metre", 
    "defence", "licence", "analyse", "realise", "organise", 
    "traveller", "jewellery", "programme", "aluminium", "plough", 
    "catalogue", "grey", "leukaemia", "tyre",  "autumn", "christmas", 
    "boxing day", "bank holiday", "easter", "good friday", "new year's day", 
    "bonfire night", "summer holidays", "half term", "london", "paris", 
    "berlin", "european union", "british", "england", "united kingdom", 
    "favourite", "theatre", "university", "headteacher", "primary school", 
    "secondary school", "european parliament", "fifa world cup", "uefa", 
    "euros", "football"
])

corpus = open("date_format_dd_mm_yyyy.txt", "r")
corpus = corpus.read()
corpus = corpus.lower()
corpus_set = corpus.split()
corpus_set = set(corpus_set)

output_file = open("q1_batool_ba07612.txt", "w", encoding='utf-8')

dates = re.findall(r'\d{1,2}\/\d{1,2}\/\d{1,4}', corpus)

for d in dates:
    if int(d[:2]) > 12:
        format = "DD/MM/YYYY"
    elif int(d[3:5]) > 12:
        format  = "MM/DD/YYYY"
    else:
        if len(corpus_set.intersection(us_spelling)) != 0:
            format = "MM/DD/YYYY"
        elif len(corpus_set.intersection(british_spelling)) != 0:
            format = "DD/MM/YYYY"
        else:
            format = "ambiguous"
    output_file.write(d + ": " + format + "\n")