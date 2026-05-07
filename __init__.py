import importlib
import traceback

node_list = [
    "image_node",
    "video_node",
    "llm_node",
    "vlm_node",
    "trainer_node",
    "upscaler_node",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Import each module independently so one bad file doesn't kill all node registrations.
for module_name in node_list:
    try:
        imported_module = importlib.import_module(f".nodes.{module_name}", __name__)
        NODE_CLASS_MAPPINGS.update(imported_module.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(imported_module.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception:
        print(f"[ComfyUI-fal-API] FAILED to load nodes.{module_name}; other modules will still load.")
        traceback.print_exc()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
