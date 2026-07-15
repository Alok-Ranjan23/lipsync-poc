from pathlib import Path

source_path = (
    Path(__file__).resolve().parent
    / "EchoMimicV3"
    / "src"
    / "pipeline_wan_fun_inpaint_audio.py"
)

original = """    else:
            batch_size = prompt_embeds.shape[0]
"""
patched = """    else:
            batch_size = len(prompt_embeds) if isinstance(prompt_embeds, list) else prompt_embeds.shape[0]
"""

source = source_path.read_text(encoding="utf-8")

if original in source:
    source_path.write_text(source.replace(original, patched), encoding="utf-8")
    print("Applied EchoMimic prompt-batch compatibility patch")
elif patched in source:
    print("EchoMimic prompt-batch compatibility patch already applied")
else:
    raise RuntimeError(f"Expected prompt batch block not found in {source_path}")
