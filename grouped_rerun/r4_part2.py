import json
import review_r4_experiments as r4
out = json.load(open(r4.OUT_PATH))
out["snli"] = r4.run_nli("snli", "stanfordnlp/snli", "validation",
                         "snli_s_2400.npy", ("premise", "hypothesis", "label"))
json.dump(out, open(r4.OUT_PATH, "w"), indent=1)
out["sick"] = r4.run_sick()
json.dump(out, open(r4.OUT_PATH, "w"), indent=1)
out["toxicity"] = r4.run_tox()
json.dump(out, open(r4.OUT_PATH, "w"), indent=1)
print("part2 done")
