from matplotlib import Path
from PIL import Image
import imagehash, json
from sklearn.model_selection import GroupShuffleSplit

IMG_DIR  = Path("D:/nosh_assignment/data/all_images")
files = sorted(p for p in IMG_DIR.iterdir()
               if p.suffix.lower() in {".jpg",".jpeg",".png"})
print("images:" , len(files))

def base_name(p):
    return p.name.split(".ref.")[0]

bases = sorted({base_name(p) for p in files})
print("unique originals:" , len(bases))

rep = {}

for p in files :
    rep.setdefault(base_name(p),p)
hashes = {b: imagehash.phash(Image.open(rep[b])) for b in bases}

parent = {b:b for b in bases}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

for i,a in enumerate(bases):
    for b in bases[i+1:]:
        if hashes[a] - hashes[b] <= 5:
            parent[find(a)] = find(b)

groups = [find(base_name(p)) for p in files]
print("final group:", len(set(groups)))

gss = GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
dev_idx,eval_idx = next(gss.split(files,groups=groups))

dev_groups = {groups[i] for i in dev_idx}
eval_groups = {groups[i] for i in eval_idx}
assert dev_groups.isdisjoint(eval_groups),"LEAKAGE!!"

split = {"dev" : [files[i].name for i in dev_idx],
         "eval" : [files[i].name for i in eval_idx]}

json.dump(split,open("split.json","w"),indent=2)
print("dev:",len(split["dev"]),"eval:" , len(split["eval"]))

for i, a in enumerate(bases):
    for b in bases[i + 1:]:
        d = hashes[a] - hashes[b]
        if d <= 10:
            print(d, a, b)