from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    api_key: str = "supersecretkey"
    model_version: str = "0.2.0"
    model_path: str = "models/restaurant_model.onnx"
    artifacts_dir: str = "models"
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
