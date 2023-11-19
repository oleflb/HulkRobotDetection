import sys
import openvino as ov
from os import path

ov_model = ov.convert_model(sys.argv[1])
ov.save_model(ov_model, f"{path.splitext(sys.argv[1])[0]}-ov.xml")
