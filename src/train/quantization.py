from ..models.ssdnet import SSDNet
from ..models.lightning import LightningWrapper
from ..dataloader.lightningdataset import DataModule
import sys
import torch
import nncf
from onnxsim import simplify
import onnx
import openvino as ov

def main():
    image_size = (240, 320)
    input_image = torch.randn((1, 3, *image_size))

    dataloader = DataModule(image_size, num_workers=12, batch_size=1)
    dataloader.setup("fit")
    val_data = dataloader.val_dataloader()

    def transform_fn(data_item):
        image_tensor = data_item[0]
        return torch.stack(image_tensor)


    ov_model = ov.convert_model(sys.argv[1], example_input=input_image)
    # model = model.model

    quantization_dataset = nncf.Dataset(val_data, transform_fn)
    quant_ov_model = nncf.quantize(ov_model, quantization_dataset)

    ov.save_model(quant_ov_model, "quantized_model.xml")

if __name__ == "__main__":
    main()
