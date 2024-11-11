quality_definitions = {
    "AspectRatio": {"good": (1, 100), "ok": (100, 1000)},
    "MeshExpansion": {"good": (1, 20), "ok": (20, 40)},
    "Skewness": {"good": (0, 0.5), "ok": (0.5, 0.75)},
    # Add more mesh quality definitions as needed
}


def classify_mesh_quality(quality_name, value_array):
    definitions = quality_definitions.get(quality_name)
    if definitions is None:
        return "Undefined Quality"

    good_range = definitions["good"]
    ok_range = definitions["ok"]

    if all((good_range[0] <= value_array) * (value_array <= good_range[1])):
        print("[ntrfc info] meshquality: GOOD")
        return True
    elif any(ok_range[1] <= value_array):
        print("[ntrfc info] meshquality: BAD")
        return False
    elif any((ok_range[0] <= value_array) * (value_array <= ok_range[1])):
        print("[ntrfc info] meshquality: OK")
        return True
    else:
        print("[ntrfc info] Undefined Quality")
        return False
