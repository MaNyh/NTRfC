quality_definitions = {
    "AspectRatio": {"good": (1, 100), "ok": (100, 1000)},
    "MeshExpansion": {"good": (1, 20), "ok": (20, 40)},
    "Skewness": {"good": (0, 0.5), "ok": (0.5, 0.75)},
    "DeltaXPlus": {"good": (0, 25), "ok": (25, 50)},
    "DeltaYPlus": {"good": (0, 1), "ok": (1, 4)},
    "DeltaZPlus": {"good": (0, 12.5), "ok": (12.5, 25)},
    # Add more mesh quality definitions as needed
}


def classify_mesh_quality(quality_name, value_array):
    definitions = quality_definitions.get(quality_name)
    if definitions is None:
        return "Undefined Quality"

    good_range = definitions["good"]
    ok_range = definitions["ok"]

    if all((good_range[0] <= value_array) * (value_array <= good_range[1])):
        print(f"[ntrfc info] meshquality {quality_name}: GOOD")
        return True
    elif any(ok_range[1] <= value_array):
        print(f"[ntrfc info] meshquality {quality_name}: BAD")
        return False
    elif any((ok_range[0] <= value_array) * (value_array <= ok_range[1])):
        print(f"[ntrfc info] meshquality {quality_name}: OK")
        return True
    else:
        print(f"[ntrfc info] Undefined Quality {quality_name}")
        return False
