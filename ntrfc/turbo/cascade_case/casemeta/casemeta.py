class CaseMeta:
    def __init__(self, case_root_directory, case_name=None):
        self.case_root_directory = case_root_directory
        self.case_name = case_name

        self.velocity_name = None
        self.meanvelocity_name = None
        self.density_name = None
        self.meandensity_name = None
        self.pressure_name = None
        self.meanpressure_name = None
        self.temperature_name = None
        self.meantemperature_name = None
        self.turbulentkineticenergy_name = None
        self.meanturbulentkineticenergy_name = None
        self.mach_name = None

        self.dynamic_viscosity = 1
        self.kappa = 1.4

    def casevariables(self, var):
        if var == "velocity":
            return self.velocity_name
        elif var == "meanvelocity":
            return self.meanvelocity_name
        elif var == "density":
            return self.density_name
        elif var == "meandensity":
            return self.meandensity_name
        elif var == "pressure":
            return self.pressure_name
        elif var == "meanpressure":
            return self.meanpressure_name
        elif var == "temperature":
            return self.temperature_name
        elif var == "meantemperature":
            return self.meantemperature_name
        elif var == "turbulentkineticenergy":
            return self.turbulentkineticenergy_name
        elif var == "meanturbulentkineticenergy":
            return self.meanturbulentkineticenergy_name
        elif var == "mach":
            return self.mach_name
        else:
            return f"Undefined variable: {var}"
