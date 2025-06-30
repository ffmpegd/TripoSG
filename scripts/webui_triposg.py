import tempfile
from typing import Any, Union

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from scripts.briarmbg import BriaRMBG
from scripts.inference_triposg import run_triposg


def load_models(device: str, dtype: torch.dtype):
    triposg_dir = "pretrained_weights/TripoSG"
    rmbg_dir = "pretrained_weights/RMBG-1.4"
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_dir)

    rmbg_net = BriaRMBG.from_pretrained(rmbg_dir).to(device)
    rmbg_net.eval()

    pipe = TripoSGPipeline.from_pretrained(triposg_dir).to(device, dtype)
    return pipe, rmbg_net


def infer(
    image: Union[str, Image.Image],
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    faces: int,
    pipe: Any,
    rmbg_net: Any,
):
    mesh = run_triposg(
        pipe,
        image_input=image,
        rmbg_net=rmbg_net,
        seed=seed,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        faces=faces,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    mesh.export(tmp.name)
    tmp.close()
    return tmp.name, tmp.name


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe, rmbg_net = load_models(device, dtype)

    with gr.Blocks() as demo:
        gr.Markdown("# TripoSG Web UI")
        with gr.Row():
            image_input = gr.Image(type="filepath", label="Input Image")
            model_out = gr.Model3D()
        download_file = gr.File(label="Download GLB")
        seed = gr.Number(value=42, label="Seed")
        steps = gr.Slider(10, 100, value=50, step=1, label="Inference Steps")
        guidance = gr.Slider(0.0, 15.0, value=7.0, step=0.1, label="Guidance Scale")
        faces_input = gr.Number(value=-1, label="Simplify Faces (-1 for none)")
        run_btn = gr.Button("Generate")

        run_btn.click(
            infer,
            inputs=[image_input, seed, steps, guidance, faces_input, pipe, rmbg_net],
            outputs=[model_out, download_file],
        )

    demo.launch()


if __name__ == "__main__":
    main()
