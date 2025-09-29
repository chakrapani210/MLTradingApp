from typing import Dict, List, Optional, Any

class ModelManager:
    def __init__(self, config_path: Optional[str] = None):
        self.models = {}
        self.active_model = None
        self.config_path = config_path

    def register_model(self, name: str, model: Any, metadata: Dict[str, Any] = None):
        self.models[name] = {
            "model": model,
            "metadata": metadata or {}
        }
        if self.active_model is None:
            self.active_model = name

    def get_model(self, name: Optional[str] = None) -> Optional[Any]:
        if name is None:
            name = self.active_model
        if name and name in self.models:
            return self.models[name]["model"]
        return None

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    def get_active_model(self) -> Optional[str]:
        return self.active_model

    def set_active_model(self, name: str) -> bool:
        if name in self.models:
            self.active_model = name
            return True
        return False

def get_model_manager() -> ModelManager:
    return ModelManager()
