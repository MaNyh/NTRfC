from ntrfc.turbo.cascade_case.casemeta.casemeta import CaseMeta


def test_casemeta(tmp_path):
    case_meta = CaseMeta(tmp_path)
    case_meta.velocity_name = "U"
    case_meta.meanvelocity_name = "Umean"
    case_meta.density_name = "rho"
    case_meta.meandensity_name = "rhoMean"
    case_meta.pressure_name = "p"
    case_meta.meanpressure_name = "pMean"
    case_meta.temperature_name = "T"
    case_meta.meantemperature_name = "Tmean"
    case_meta.turbulentkineticenergy_name = "k"
    case_meta.meanturbulentkineticenergy_name = "kMean"
    case_meta.mach_name = "Mach"

    assert case_meta.casevariables("velocity") == "U"
    assert case_meta.casevariables("meanvelocity") == "Umean"
    assert case_meta.casevariables("density") == "rho"
    assert case_meta.casevariables("meandensity") == "rhoMean"
    assert case_meta.casevariables("pressure") == "p"
    assert case_meta.casevariables("meanpressure") == "pMean"
    assert case_meta.casevariables("temperature") == "T"
    assert case_meta.casevariables("meantemperature") == "Tmean"
    assert case_meta.casevariables("turbulentkineticenergy") == "k"
    assert case_meta.casevariables("meanturbulentkineticenergy") == "kMean"
    assert case_meta.casevariables("mach") == "Mach"
    assert case_meta.casevariables("undefined") == "Undefined variable: undefined"
