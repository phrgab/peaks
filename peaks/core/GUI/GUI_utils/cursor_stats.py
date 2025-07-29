def _parse_norm_emission_cursor_stats(da, norm_emission_dict):
    cursor_text = ""
    if norm_emission_dict:
        cursor_text += "Normal emission: "
        # Try and parse local names
        for dim in norm_emission_dict.copy():
            local_name = _get_nested_attr(
                da.metadata, f"manipulator.{dim}.local_name"
            ) or _get_nested_attr(da.metadata, f"analyser.deflector.{dim}.local_name")
            if local_name:
                norm_emission_dict[f"{dim} [{local_name}]"] = norm_emission_dict.pop(dim)

        # Set the normal emission text
        cursor_text += ", ".join([f"{k}={v:.2f}" for k, v in norm_emission_dict.items()])
    else:
        cursor_text = "<br>"
    cursor_text += "<hr>"

    return cursor_text


def _get_nested_attr(obj, attr_path):
    """Retrieve a nested attribute from an object using a dot-separated path."""
    attrs = attr_path.split(".")
    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            return None
    return obj
