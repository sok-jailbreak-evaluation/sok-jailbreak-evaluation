"""Text normalization used by the TriPlay-RL reward."""

def normalize_text(generated_content: str) -> str:
    """
    Normalize text by removing extra spaces and newlines.
    
    Args:
        generated_content: Input text string
    
    Returns:
        Normalized text string
    """
    if generated_content.find('</think>') != -1:
        generated_content = generated_content[generated_content.find('</think>') + len('</think>'):].strip()
    if generated_content.find('```json') != -1:
        generated_content = generated_content[generated_content.find('```json') + len('```json'):].strip()
    if generated_content.find('```') != -1:
        generated_content = generated_content[:generated_content.find('```')].strip()
    return generated_content.strip()
