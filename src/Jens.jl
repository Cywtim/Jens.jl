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
    # # include("utils/LensLOS.jl")       # 旧路径: 已移至 models/lens/
    include("models/lens/LensLOS.jl")     # LOS tidal matrix
    include("utils/LensPSF.jl")           # PSF must precede Generator + Solver
    include("utils/LensSolver.jl")        # lens equation solver
    include("models/light/LightModel.jl")  # AbstractLight types needed by Generator
    include("core/LensGenerator.jl")      # needs LensPSF + LensSolver + LightModel
    include("utils/LensNoise.jl")
    include("utils/WFC3.jl")
    include("utils/MGE.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Models
    # ═══════════════════════════════════════════════════════════════
    include("models/lens/LensModel.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Plotting
    # ═══════════════════════════════════════════════════════════════
    include("plotting/LensPlots.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Exports
    # ═══════════════════════════════════════════════════════════════
    export LensUtils, LensBase
    export LensFITS
    export LensGenerator, LensCosmo, LensLOS, LensSolver, LensNoise, LensPSF, WFC3
    export LensModel, LightModel
    export LensPlots

end
