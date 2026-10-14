import os
from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

def export_and_quantize_model(model_id="ai4bharat/indictrans2-indic-indic-dist-320M", save_dir="models_onnx/indictrans2_quantized"):
    """
    Exports PyTorch Hugging Face model to ONNX and applies dynamic INT8 quantization
    for low-end offline Android / Edge deployment.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"[1/3] Exporting {model_id} to ONNX format...")
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    temp_dir = f"{save_dir}_raw"
    ort_model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)

    print("[2/3] Applying Dynamic INT8 Quantization for Low-Cost Devices...")
    quantizer = ORTQuantizer.from_pretrained(temp_dir)
    dqconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
    
    quantizer.quantize(save_dir=save_dir, quantization_config=dqconfig)
    tokenizer.save_pretrained(save_dir)
    print(f"[3/3] Quantized Offline Model saved to: {save_dir}")

if __name__ == "__main__":
    export_and_quantize_model()