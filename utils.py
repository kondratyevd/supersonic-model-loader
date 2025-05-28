import re
import socket

def find_free_port() -> int:
    """
    Find a free port on the local machine.
    
    Returns:
        int: A free port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def parse_model_name(full_name: str) -> tuple[str, str]:
    """
    Parse full model name into name and version.
    Only splits on '-v' if it's at the end of the name.
    
    Examples:
        "my-model-v1" -> ("my-model", "1")
        "model-with-v2-in-name-v1" -> ("model-with-v2-in-name", "1")
        "model-without-version" -> ("model-without-version", "1")
    """
    # Match '-v' followed by numbers at the end of the string
    match = re.match(r'^(.*?)-v(\d+)$', full_name)
    if match:
        return match.group(1), match.group(2)
    return full_name, "1"

def format_model_label(model_name: str) -> str:
    """Format model name into label key format"""
    model_name = escape_model_name(model_name)
    return f"sonic.model.loaded/{model_name}"

def escape_model_name(model_name: str) -> str:
    """Escape model name for use as a label"""
    return model_name.lower().replace("_", "-")