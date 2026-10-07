"""Rewrite INT8 mlpackage as fp16 (constexpr_affine_dequantize -> const) to test whether int8 weights crash the GPU path."""
import coremltools as ct, time
from coremltools.converters.mil import mil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.frontend.milproto.load import load as load_milproto
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import numpy as np

src = "model/LAMReconstruct_int8.mlpackage"
m = ct.models.MLModel(src, skip_model_load=True)
spec = m.get_spec()
prog = load_milproto(spec, specification_version=spec.specificationVersion,
                     file_weights_dir=src + "/Data/com.apple.CoreML/weights")
n = 0
for f in prog.functions.values():
    for op in list(f.operations):
        if op.op_type == "constexpr_affine_dequantize":
            val = op.materialized_val_inference().astype(np.float16)
            with f:
                pass
            blk = f
            with blk:
                new = mb.const(val=val, before_op=op, name=op.outputs[0].name + "_fp16")
            blk.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=new, no_check_var_types=True, force_replace=True)
            blk.remove_ops([op]); n += 1
print("replaced", n)
t = time.time()
out = ct.convert(prog, convert_to="mlprogram", inputs=[ct.ImageType(name="input_image", shape=(1,3,518,518), color_layout=ct.colorlayout.RGB)], compute_precision=ct.precision.FLOAT16,
                 minimum_deployment_target=ct.target.macOS14)
out.save("model/LAMReconstruct_fp16.mlpackage")
print("saved %.0fs" % (time.time() - t))
