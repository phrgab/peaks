from peaks.core.fileIO.loc_registry import (
    LOC_REGISTRY,
    locs,
)


class TestLocsRegistry:
    def test_locs_returns_set(self):
        result = locs()
        assert isinstance(result, set)
        assert len(result) > 0

    def test_registry_contains_known_loaders(self):
        registered = locs()
        expected_loaders = {
            "CLF_Artemis",
            "Diamond_I05_ARPES",
            "Diamond_I05_Nano-ARPES",
            "Elettra_APE_LE",
            "FeSuMa",
            "MAXIV_Bloch_A",
            "MBS",
            "NetCDF",
            "SES",
            "Soleil_Casiopee_ARPES",
            "Specs",
            "StA_MBS",
            "StA_Phoibos",
            "Zarr",
            "cif",
            "ibw",
        }
        missing = expected_loaders - registered
        assert missing == set(), f"Loaders not registered: {missing}"

    def test_registry_is_dict(self):
        assert isinstance(LOC_REGISTRY, dict)

    def test_registry_values_are_classes(self):
        for name, loader in LOC_REGISTRY.items():
            assert isinstance(loader, type), f"{name} is not a class"
