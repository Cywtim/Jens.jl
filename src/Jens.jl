module Jens

    # ═══════════════════════════════════════════════════════════════
    #  Core — LensUtils included ONCE, shared by all submodules
    # ═══════════════════════════════════════════════════════════════
    include("core/LensUtils.jl")
    include("core/LensBase.jl")

    # ═══════════════════════════════════════════════════════════════
    #  I/O
    # ═══════════════════════════════════════════════════════════════
    include("io/LensFITS.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Utilities
    # ═══════════════════════════════════════════════════════════════
    include("utils/LensGenerator.jl")
    include("utils/LensCosmo.jl")
    include("utils/LensNoise.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Models
    # ═══════════════════════════════════════════════════════════════
    include("models/lens/LensModel.jl")
    include("models/light/LightModel.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Plotting
    # ═══════════════════════════════════════════════════════════════
    include("plotting/LensPyPlot.jl")

end
