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
        elif isinstance(v, (int, float)):
            return v * ureg("")
        elif isinstance(v, np.ndarray):
            return v * ureg("")  # Handle ndarray without units
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
class BaseMetadataUnitsModel(BaseModel):
    """Generalized model to store metadata, with value as the primary attribute."""

    value: Quantity  # Primary value can be a scalar or ndarray
    model_config = ConfigDict(json_encoders={pint.Quantity: _quantity_encoder})
