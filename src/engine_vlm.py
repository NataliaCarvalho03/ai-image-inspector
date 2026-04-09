import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer


class VLMEngine:
    MAX_NEW_TOKENS = 256
    HIDDEN_SIZE    = 2048

    def __init__(self, model_dir="models/moondream"):
        self.model_dir    = model_dir
        self.vision_sess  = None
        self.text_sess    = None
        self.tokenizer    = None
        self.embed_tokens = None

        vision_path  = os.path.join(model_dir, "onnx", "vision_encoder_q4.onnx")
        decoder_path = os.path.join(model_dir, "onnx", "decoder_model_merged_bnb4.onnx")
        embed_path   = os.path.join(model_dir, "onnx", "model_bnb4.onnx")
        tok_path     = os.path.join(model_dir, "tokenizer.json")

        for label, path in [("Vision encoder", vision_path),
                             ("Decoder",        decoder_path),
                             ("Embed model",    embed_path),
                             ("Tokenizer",      tok_path)]:
            if not os.path.exists(path):
                print(f"Warning: {label} not found at {path}")
                return

        providers = self._get_providers()
        print(f"Loading Moondream2 ONNX with providers: {providers}")

        try:
            self.vision_sess = ort.InferenceSession(vision_path,  providers=providers)
            self.text_sess   = ort.InferenceSession(decoder_path, providers=providers)
            self.tokenizer   = Tokenizer.from_file(tok_path)

            self._load_embed_weights(embed_path)

            self._decoder_input_names  = [i.name for i in self.text_sess.get_inputs()]
            self._decoder_output_names = [o.name for o in self.text_sess.get_outputs()]

            self._kv_inputs  = [n for n in self._decoder_input_names  if n.startswith("past_key_values")]
            self._kv_outputs = [n for n in self._decoder_output_names if n.startswith("present")]

            self._kv_meta = {
                i.name: i.shape
                for i in self.text_sess.get_inputs()
                if i.name.startswith("past_key_values")
            }

            print(f"Moondream2 loaded! KV layers: {len(self._kv_inputs) // 2}")
        except Exception as e:
            print(f"Error loading Moondream2: {e}")

    def _load_embed_weights(self, embed_path: str):
        import onnx
        m = onnx.load(embed_path)
        for init in m.graph.initializer:
            if init.name == "model.embed_tokens.weight":
                arr = np.frombuffer(init.raw_data, dtype=np.float32).copy()
                self.embed_tokens = arr.reshape(list(init.dims))
                print(f"Loaded embed_tokens: {self.embed_tokens.shape}")
                return
        raise RuntimeError("embed_tokens weight not found in model_bnb4.onnx")

    def is_loaded(self) -> bool:
        return all([self.vision_sess, self.text_sess, self.tokenizer, self.embed_tokens is not None])

    def generate_description(self, image: "Image.Image", prompt: str = "Describe this image.") -> str:
        if not self.is_loaded():
            return "Error: Model not loaded."

        try:
            image_features = self._encode_image(image)

            full_prompt = f"\n\nQuestion: {prompt}\n\nAnswer:"
            input_ids   = self.tokenizer.encode(full_prompt).ids

            generated_ids = self._generate(input_ids, image_features)

            return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        except Exception as e:
            return f"Inference error: {e}"

    def _encode_image(self, image: "Image.Image") -> np.ndarray:
        img = image.convert("RGB").resize((378, 378))
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        arr  = (arr - mean) / std
        arr  = arr.transpose(2, 0, 1)[np.newaxis]

        vision_input   = self.vision_sess.get_inputs()[0].name
        image_features = self.vision_sess.run(None, {vision_input: arr})[0]
        return image_features

    def _embed(self, token_ids: list) -> np.ndarray:
        indices = np.array(token_ids, dtype=np.int64)
        return self.embed_tokens[indices][np.newaxis]

    def _init_past_key_values(self) -> dict:
        pkv = {}
        for inp in self.text_sess.get_inputs():
            if not inp.name.startswith("past_key_values"):
                continue
            shape = []
            for idx, d in enumerate(inp.shape):
                if isinstance(d, str):
                    shape.append(1 if "batch" in d else 0)
                else:
                    shape.append(d)
            shape[2] = 0
            pkv[inp.name] = np.zeros(shape, dtype=np.float32)
        return pkv

    def _generate(self, input_ids: list, image_features: np.ndarray) -> list:
        end_token = self.tokenizer.token_to_id("<END>")
        if end_token is None:
            end_token = self.tokenizer.token_to_id("</s>")

        text_embeds   = self._embed(input_ids)
        inputs_embeds = np.concatenate([image_features, text_embeds], axis=1)

        img_len  = image_features.shape[1]
        txt_len  = text_embeds.shape[1]
        total    = img_len + txt_len

        past_kv  = self._init_past_key_values()
        generated = []

        for step in range(self.MAX_NEW_TOKENS):
            if step == 0:
                cur_embeds = inputs_embeds
                cur_len    = total
            else:
                next_tok   = generated[-1]
                cur_embeds = self._embed([next_tok])
                cur_len    = 1

            past_len = 0 if step == 0 else (total + step - 1)
            attn_len = past_len + cur_len
            attn     = np.ones((1, attn_len), dtype=np.int64)

            if step == 0:
                pos_ids = np.arange(cur_len, dtype=np.int64)[np.newaxis]
            else:
                pos_ids = np.array([[past_len + cur_len - 1]], dtype=np.int64)

            feeds = {
                "inputs_embeds":  cur_embeds,
                "attention_mask": attn,
                "position_ids":   pos_ids,
            }
            feeds.update(past_kv)
            feeds = {k: v for k, v in feeds.items() if k in self._decoder_input_names}

            outputs = self.text_sess.run(self._decoder_output_names, feeds)
            results = dict(zip(self._decoder_output_names, outputs))

            logits  = results["logits"]
            next_id = int(np.argmax(logits[0, -1, :]))

            if next_id == end_token:
                break

            generated.append(next_id)

            past_kv = {}
            for out_name, inp_name in zip(self._kv_outputs, self._kv_inputs):
                if out_name in results:
                    past_kv[inp_name] = results[out_name]

        return generated

    @staticmethod
    def _get_providers() -> list:
        available = ort.get_available_providers()
        for provider in ["CUDAExecutionProvider", "CoreMLExecutionProvider",
                         "DirectMLExecutionProvider"]:
            if provider in available:
                return [provider, "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


if __name__ == "__main__":
    engine = VLMEngine(model_dir="models/moondream")
    image  = Image.open("/home/natalia/Documents/llm-tests/ai-image-inspector/0c37b810-f7cc-11ec-9452-478d987e1a52.jpg")
    print(engine.generate_description(image))