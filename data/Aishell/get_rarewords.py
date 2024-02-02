import json

setname = "dev"

with open("{}.json".format(setname), encoding='utf-8') as fin:
    data = json.load(fin)

with open("Blist/rareword.dev.txt", encoding='utf-8') as fin:
    rarewords = [word.strip() for word in fin]

for uttname, utt in data.items():
    uttKB = []
    words = utt["words"].split()
    for word in rarewords:
        if word in utt["words"]:
            uttKB.append(word)
    data[uttname]["blist"] = uttKB

with open("{}_full.json".format(setname), "w", encoding='utf-8') as fout:
    json.dump(data, fout, ensure_ascii=False, indent=4)


