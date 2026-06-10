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
    # # include("core/LensGenerator.jl")  # 旧顺序: Generator 在 Cosmo 前
    # 改为 Cosmo 先加载, 因为 LensGenerator 依赖 LensCosmo.lens_distance_ratio
    include("utils/LensCosmo.jl")
    include("core/LensGenerator.jl")
    include("utils/LensNoise.jl")
    include("utils/LensPSF.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Models
    # ═══════════════════════════════════════════════════════════════
    include("models/lens/LensModel.jl")
    include("models/light/LightModel.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Plotting
    # ═══════════════════════════════════════════════════════════════
    include("plotting/LensPlots.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Exports
    # ═══════════════════════════════════════════════════════════════
    export LensUtils, LensBase
    export LensFITS
    export LensGenerator, LensCosmo, LensNoise, LensPSF
    export LensModel, LightModel
    export LensPlots

end
