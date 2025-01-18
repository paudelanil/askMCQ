import yaml
from langchain_openai import OpenAIEmbeddings

def load_config():
    """
    Loads configuration from config.yaml.
    """
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def get_embedding_function():
    """
    Returns an OpenAI embedding function.
    """
    config = load_config()
    return OpenAIEmbeddings(
        model=config["openai"]["embedding_model"],
        openai_api_key=config["openai"]["api_key"]
    )