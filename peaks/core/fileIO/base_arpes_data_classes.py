from typing import Optional, Union
from pydantic import BaseModel, create_model

from .base_data_classes import (
    BaseManipulatorDataLoader,
    BaseCryoDataLoader,
    BasePhotonDataLoader,
)
from .base_metadata_models import Quantity


class BaseARPESDataLoader(
    BasePhotonDataLoader, BaseCryoDataLoader, BaseManipulatorDataLoader
):
    """Base class for data loaders for ARPES systems. Assume a cryo-manipulator and photon source"""

    loc_name = "Default ARPES"
    dtype = "float32"
    dorder = "C"

    analyser_model = None
    analyser_type = "I"

    desired_dim_order = (
        [
            "scan_no",
            "hv",
            "temperature_sample",
            "temperature_cryostat",
        ]
        + BaseManipulatorDataLoader.desired_dim_order.copy()
        + [
            "y_scale",
            "defl_perp",
            "eV",
            "defl_par",
            "theta_par",
        ]
    )
    arpes_units = {
        "y_scale": "mm",
        "eV": "eV",
        "PE": "eV",
        "step_size": "meV",
        "theta_par": "deg",
        "entrance_slit_width": "mm",
        "dwell": "s",
        "ana_mode": None,
        "sweeps": None,
    }

    arpes_sign_conventions = {"theta_par": 1}
    arpes_name_conventions = {}
    arpes_expected_metadata = [
        "PE",
        "eV",
        "step_size",
        "sweeps",
        "dwell",
        "ana_mode",
        "entrance_slit_width",
        "entrance_slit_width_identifier",
        "analyser_type",
        "scan_type",
    ]

    class ARPESAnalyserMetadataModel(BaseModel):
        """Model to store core analyser-linked metadata."""

        class ARPESSlitMetadataModel(BaseModel):
            """Model to store slit width metadata."""

            width: Optional[Union[Quantity, str]] = None
            identifier: Optional[str] = None

        model: Optional[str] = None
        entrance_slit: Optional[ARPESSlitMetadataModel] = None

    class ARPESScanMetadataModel(BaseModel):
        """Model to store scan metadata."""

        eV: Optional[Union[Quantity, str]] = None
        step_size: Optional[Union[Quantity, str]] = None
        PE: Optional[Union[Quantity, str]] = None
        sweeps: Optional[int] = None
        dwell: Optional[Union[Quantity, str]] = None
        scan_type: Optional[str] = None

    class ARPESAnalyserAnglesMetadataModel(BaseModel):
        """Model to store analyser angles metadata."""

        polar: Optional[Union[Quantity, str]] = None
        tilt: Optional[Union[Quantity, str]] = None
        azi: Optional[Union[Quantity, str]] = None

    class ARPESDeflectorMetadataModel(BaseModel):
        """Model to store deflector metadata."""

        parallel: Optional[Union[Quantity, str]] = None
        perp: Optional[Union[Quantity, str]] = None

    @classmethod
    def create_arpes_metadata_model(cls):
        """Function to dynamically create ARPES metadata model."""
        fields = {
            "slit": (Optional[cls.ARPESSlitMetadataModel], None),
            "analyser": (Optional[cls.ARPESAnalyserMetadataModel], None),
            "scan": (Optional[cls.ARPESScanMetadataModel], None),
            "angles": (Optional[cls.ARPESAnalyserAnglesMetadataModel], None),
            "deflector": (Optional[cls.ARPESDeflectorMetadataModel], None),
        }
        return create_model("ARPESMetadataModel", **fields)

    @classmethod
    def load_metadata(cls, fname):
        """Load the metadata. Should return a dictionary `metadata_dict` mapping relevant metadata keys to values.

        See Also
        --------
        BaseDataLoader.load_metadata
        """

        metadata_dict = super().load_metadata(fname)
        metadata_dict.update(cls.load_arpes_metadata(fname))
        return metadata_dict

    @classmethod
    def load_arpes_metadata(cls, fname):
        """Load photon metadata from the file.

        Returns
        -------
        dict :
            Dictionary of values, with keys `photon_eV`, `photon_polarisation`,`photon_exit_slit`.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def apply_metadata(cls, DataArray, metadata):
        """Apply metadata to the DataArray."""
        super().apply_metadata(DataArray, metadata)

        # Populate the photon metadata model
        photon_metadata = cls.PhotonMetadataModel(
            eV=cls.apply_units_to_metadata(
                metadata.get("photon_eV"), cls.photon_units["eV"]
            ),
            polarisation=metadata.get("photon_polarisation"),
            exit_slit=cls.apply_units_to_metadata(
                metadata.get("photon_exit_slit"), cls.photon_units["exit_slit"]
            ),
        )
        DataArray.attrs.update({"photon": photon_metadata})

    # TODO Add in here methods for dealing with angles...
