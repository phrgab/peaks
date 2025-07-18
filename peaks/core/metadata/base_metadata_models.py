from typing import Optional, Union

import numpy as np
import pint
import pint_xarray
from pydantic import BaseModel, ConfigDict
from pydantic_core import core_schema

# Define the appropriate unit registry
ureg = pint_xarray.unit_registry


# Class to handle storing and validating pint Quantities in pydantic model
class Quantity(pint.Quantity):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        # Use with_info_plain_validator_function to handle both value and info
        return core_schema.with_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v, info):
        if isinstance(v, pint.Quantity):
            return v
        elif isinstance(v, dict):
            value = v.get("value")
            units = v.get("units")
            if value is not None and units is not None:
                if isinstance(value, list):  # Handle list serialization of ndarray
                    value = np.array(value)
                return value * ureg(units)
            else:
                raise ValueError(
                    'Invalid quantity dictionary. Must have "value" and "units".'
                )
        elif (
            isinstance(v, tuple)
            and len(v) == 2
            and isinstance(v[1], str)
            and isinstance(v[0], (int, float, list, np.ndarray))
        ):
            return v[0] * ureg(v[1])
        elif isinstance(v, (int, float)):
            return v * ureg("")
        elif isinstance(v, np.ndarray):
            return v * ureg("")
        else:
            raise TypeError(f"Invalid type for Quantity: {type(v)}")

    def __repr__(self):
        return f"{self.magnitude} {self.units}"


# Handle serialising
def _quantity_encoder(quantity: pint.Quantity):
    if isinstance(quantity.magnitude, np.ndarray):
        # Convert ndarray to list for JSON serialization
        return {"value": quantity.magnitude.tolist(), "units": str(quantity.units)}
    return {"value": quantity.magnitude, "units": str(quantity.units)}


# Base class for passing a pint Quantity
class BaseMetadataModel(BaseModel):
    """Generalized model to store metadata, allowing serialising pint.Quantity objects."""

    model_config = ConfigDict(
        json_encoders={pint.Quantity: _quantity_encoder}, validate_assignment=True
    )


class BaseScanMetadataModel(BaseMetadataModel):
    """Model to store basic scan identifier metadata."""

    name: str
    filepath: str
    loc: str
    timestamp: str
    scan_command: Optional[str] = None


class NamedAxisMetadataModel(BaseMetadataModel):
    """Based model to store metadata for a single named axis.

    Attributes
    ----------
    local_name : Optional[str]
        The name of the axis.
    value : Optional[Union[Quantity, str, None]]
        The value of the axis.

    Methods
    -------
    set(value)
        Set the value of the axis.
    get()
        Get the value of the axis.

    local_name : Optional[str]
        The name of the axis.
    value : Optional[Union[Quantity, str, None]]
        The value of the axis.

    """

    local_name: Optional[str] = None
    value: Optional[Union[str, Quantity]] = None


# Define the manipulator metadata models
class AxisMetadataModelWithReference(NamedAxisMetadataModel):
    """Base model to store metadata for a single axis (e.g. manipulator axis) where a reference value is needed.

    Attributes
    ----------
    name : Optional[str]
        The local name of the axis on the actual system manipulator.
    value : Optional[Union[Quantity, str, None]]
        The value of the axis. If an array (e.g. for movement during the scan), this should return a string that
        describes the axis movement in the form x0:x_step:x1.
    reference_value : Optional[Union[Quantity, None]]
        The reference value of the axis. Supplied as None on initial load, but can be used later for keeping track
        e.g. of normal emission values.

    Methods
    -------
    In addition to the methods of NamedAxisMetadataModel, this class also has the following methods:
    set_reference(value)
        Set the reference value of the axis.
    """

    reference_value: Optional[Quantity] = None


# Define the temperature metadata models
class TemperatureMetadataModel(BaseMetadataModel):
    """Model to store temperature metadata."""

    sample: Optional[Union[str, Quantity]] = None
    cryostat: Optional[Union[str, Quantity]] = None
    shield: Optional[Union[str, Quantity]] = None
    setpoint: Optional[Union[str, Quantity]] = None


# Define the photon metadata models
class PhotonMetadataModel(BaseMetadataModel):
    """Model to store photon-linked metadata."""

    hv: Optional[Union[str, Quantity]] = None
    polarisation: Optional[Union[str, int, float]] = None
    exit_slit: Optional[Union[str, Quantity]] = None


class PumpPhotonMetadataModel(BaseMetadataModel):
    """Model to store pump beam metadata for pump-probe experiments."""

    hv: Optional[Union[str, Quantity]] = None
    polarisation: Optional[Union[str, int, float]] = None
    power: Optional[Union[str, Quantity]] = None
    delay: Optional[Union[str, Quantity]] = None
    t0_position: Optional[Union[str, Quantity]] = None


class ARPESSlitMetadataModel(BaseMetadataModel):
    """Model to store slit width metadata."""

    width: Optional[Union[str, Quantity]] = None
    identifier: Optional[str] = None


class ARPESAnalyserMetadataModel(BaseMetadataModel):
    """Model to store core analyser-linked metadata."""

    model: Optional[str] = None
    slit: ARPESSlitMetadataModel = ARPESSlitMetadataModel()


class ARPESScanMetadataModel(BaseMetadataModel):
    """Model to store scan metadata."""

    eV: Optional[Union[str, Quantity]] = None
    step_size: Optional[Union[str, Quantity]] = None
    PE: Optional[Union[str, Quantity]] = None
    sweeps: Optional[int] = None
    dwell: Optional[Union[str, Quantity]] = None
    lens_mode: Optional[str] = None
    acquisition_mode: Optional[str] = None
    eV_type: Optional[str] = None


class ARPESAnalyserAnglesMetadataModel(BaseMetadataModel):
    """Model to store analyser angles metadata.
    This is a general definiton which allows distinguishing analyser type (slit parallel or perp to slit)
    as well as allowing for the general case where the analyser can move."""

    polar: Optional[Union[str, Quantity]] = None
    tilt: Optional[Union[str, Quantity]] = None
    azi: Optional[Union[str, Quantity]] = None


class ARPESDeflectorMetadataModel(BaseMetadataModel):
    """Model to store deflector metadata."""

    parallel: NamedAxisMetadataModel = NamedAxisMetadataModel()
    perp: NamedAxisMetadataModel = NamedAxisMetadataModel()


class ARPESMetadataModel(BaseMetadataModel):
    """Model to store ARPES metadata."""

    analyser: ARPESAnalyserMetadataModel = ARPESAnalyserMetadataModel()
    scan: ARPESScanMetadataModel = ARPESScanMetadataModel()
    angles: ARPESAnalyserAnglesMetadataModel = ARPESAnalyserAnglesMetadataModel()
    deflector: ARPESDeflectorMetadataModel = ARPESDeflectorMetadataModel()


class ARPESCalibrationModel(BaseMetadataModel):
    EF_correction: Optional[
        Union[
            float,
            dict,
            int,
        ]
    ] = None
    V0: Optional[Union[str, Quantity]] = None
