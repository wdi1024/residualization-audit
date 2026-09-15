import json
import review_r4_experiments as r4
out = json.load(open(r4.OUT_PATH))
out["toxicity"] = r4.run_tox()
json.dump(out, open(r4.OUT_PATH, "w"), indent=1)
print("tox redone")
