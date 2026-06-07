from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "local"
    log_level: str = "INFO"
    service_port: int = 8000

    model_name: str = "BAAI/bge-m3"
    use_fp16: bool = True
    devices: str = "cpu"
    model_batch_size: int = 12
    max_length: int = 8192
    # Cap simultaneous encode() calls — CPU inference doesn't parallelize well;
    # torch already uses multiple threads internally via torch.set_num_threads.
    max_concurrent_inferences: int = 2
    # 0 => let torch decide; pin on noisy multi-tenant hosts; ignored on GPU
    torch_num_threads: int = 0
    max_texts_per_request: int = 128

    # Static API key for request authentication. Empty string disables auth (local dev).
    api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
