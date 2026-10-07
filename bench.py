import sys, time, numpy as np
import coremltools as ct
from PIL import Image
u = sys.argv[1]
t = time.time()
m = ct.models.MLModel((sys.argv[2] if len(sys.argv)>2 else "model/LAMReconstruct_int8.mlpackage"), compute_units=ct.ComputeUnit[u])
print(u, "load %.1fs" % (time.time() - t), flush=True)
img = Image.open("/private/tmp/claude-501/-Users-evindrews-Projects/259805a7-e79e-45ef-a5bc-42800987346d/images/1.webp").convert("RGB").resize((518, 518))
for i in range(3):
    t = time.time(); o = m.predict({"input_image": img})["gaussian_attributes"]
    print(u, "predict %d: %.2fs" % (i, time.time() - t), flush=True)
np.save(f"/tmp/claude-501/-Users-evindrews-Projects/259805a7-e79e-45ef-a5bc-42800987346d/scratchpad/out_{u}.npy", np.asarray(o, dtype=np.float32)[0])
