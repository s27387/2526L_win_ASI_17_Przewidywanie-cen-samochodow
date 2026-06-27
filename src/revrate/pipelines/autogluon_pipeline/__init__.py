try:
    from .pipeline import create_pipeline
except ImportError:
    pass

__all__ = ["create_pipeline"]
