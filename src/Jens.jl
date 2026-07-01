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
    #  System
    # ═══════════════════════════════════════════════════════════════
    include("core/LensSystem.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Models
    # ═══════════════════════════════════════════════════════════════
    include("models/lens/LensModel.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Plotting
    # ═══════════════════════════════════════════════════════════════
    include("plotting/LensPlots.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Sampling
    # ═══════════════════════════════════════════════════════════════
    include("sampling/LensMH.jl")      # standalone adaptive MH (no deps)
    include("sampling/LensHMC.jl")     # HMC/NUTS via AdvancedHMC + FiniteDiff
    include("sampling/LensSample.jl")  # two-stage: HMC explore → MH refine
    include("sampling/LensTuring.jl")  # Turing @model wrappers

    # ═══════════════════════════════════════════════════════════════
    #  Exports
    # ═══════════════════════════════════════════════════════════════
    export LensUtils, LensBase
    export LensFITS
    export LensGenerator, LensCosmo, LensLOS, LensSolver, LensNoise, LensPSF, WFC3
    export LensSystem
    export LensModel, LightModel
    export LensPlots
    export LensTuring, LensMH, LensHMC, LensSample

    # ═══════════════════════════════════════════════════════════════
    #  GPU convenience (filled by ext/JensCUDA.jl when CUDA loaded)
    # ═══════════════════════════════════════════════════════════════
    """
        gpu_grid(; pix_n=256, pix_size=Float32(0.09)) -> GridGPU

    Create a GPU grid.  Requires `using CUDA` beforehand.
    When CUDA is not loaded, calling this gives a MethodError.

    # Example
        using Jens, CUDA
        grid = gpu_grid(pix_n=256, pix_size=Float32(0.04))
    """
    function gpu_grid end
    export gpu_grid

end
