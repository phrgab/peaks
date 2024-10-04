""" Registry for file loaders and locations. """

LOC_REGISTRY = {}


def register_loader(loader_class):
    """Decorator to register a loader class in the LOC_REGISTRY."""
    LOC_REGISTRY[loader_class.loc_name] = loader_class
    return loader_class
