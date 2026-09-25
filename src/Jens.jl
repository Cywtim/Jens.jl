module Jens

    # ═══════════════════════════════════════════════════════════════
    #  Global precision — change this line to switch Float32 ↔ Float64
    # ═══════════════════════════════════════════════════════════════
    const JFloat = Float32   # Float32 (GPU-friendly) or Float64 (high-precision)

    # ═══════════════════════════════════════════════════════════════
    #  Core — LensUtils included ONCE, shared by all submodules
    # ═══════════════════════════════════════════════════════════════
    include("core/LensUtils.jl")
    include("core/LensBase.jl")

    # ═══════════════════════════════════════════════════════════════
    #  Utilities
    # ═══════════════════════════════════════════════════════════════
    # # include("core/LensGenerator.jl")  # 旧顺序: Generator 在 Cosmo 前
    # 改为 Cosmo 先加载, 因为 LensGenerator 依赖 LensCosmo.lens_distance_ratio
    include("utils/LensConstants.jl")
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
    include("utils/LensMask.jl")
    include("core/LensObservation.jl")   # needs Grid + LensNoise + AbstractPSF

    # ═══════════════════════════════════════════════════════════════
    #  I/O  (after Observation so LensFITS can produce Observation)
    # ═══════════════════════════════════════════════════════════════
    include("io/LensFITS.jl")
    include("io/LensFITSIO.jl")

    # ═══════════════════════════════════════════════════════════════
    #  System
    # ═══════════════════════════════════════════════════════════════
    include("core/LensSystem.jl")
    include("utils/LensTimeDelay.jl")   # needs ForwardModel from LensSystem
    using .TimeDelay: LensTimeDelay, image_time_delays
    include("utils/LensPointLikelihood.jl")  # point-source χ² (images + delays)
    include("utils/LensAdaptiveGrid.jl")     # adaptive sub-sampling near caustics
    include("models/light/LensShapelet.jl")  # shapelet-basis source reconstruction
    include("utils/LensMeshRefine.jl")       # quad-tree adaptive grid (mass & source)

    # ═══════════════════════════════════════════════════════════════
    #  Models
    # ═══════════════════════════════════════════════════════════════
    include("models/lens/LensModel.jl")
    include("models/lens/LensMassRecon.jl")  # quad-tree mass-field reconstruction
    include("models/lens/QuadTreeFit.jl")    # image-plane fit w/ quadtree residual

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
    include("sampling/LensPSO.jl")     # PSO global search → M-H refinement

    # ═══════════════════════════════════════════════════════════════
    #  Exports
    # ═══════════════════════════════════════════════════════════════
    export LensUtils, LensBase
    export LensGenerator, LensCosmo, LensConstants, LensLOS, LensSolver, LensNoise, LensPSF, LensMask, LensObservation, WFC3
    export LensFITS
    export LensFITSIO
    export LensSystem
    export LensTimeDelay, image_time_delays
    export LensPointLikelihood, LensAdaptiveGrid, LensShapelet, LensMeshRefine
    export LensModel, LightModel, LensMassRecon, QuadTreeFit
    export LensPlots
    export LensTuring, LensMH, LensHMC, LensSample, LensPSO

    # ═══════════════════════════════════════════════════════════════
    #  Quad-tree lens stack — semantic identity (Scheme A, no move)
    #
    #  The adaptive quad-tree lens is ONE coherent stack spread across
    #  three modules (kept separate for dependency hygiene, not by
    #  accident).  Entry points:
    #
    #    LensMeshRefine — bottom mesh: QuadLeaf / QuadTree (pure
    #      adaptive-grid data structures; shared with source-plane mesh
    #      work, NO lens semantics).
    #    LensMassRecon  — the quad-tree LENS ENGINE (de-facto submodule):
    #      MassField (adaptive κ field, P0/P1/P2, ROI-capped refine),
    #      field_value/mass_*, quadtree_deflection (+multipole &
    #      Barnes–Hut), quadtree_potential, quadtree_hessian,
    #      QuadTreeLens <: AbstractLens.
    #    QuadTreeFit    — image-plane fitness: quadtree_image_logp
    #      (scheme 1b: analytic base + quadtree residual + BH render),
    #      HybridQuadLens.
    #
    #  Typical flow: MeshRefine → LensMassRecon reconstructs κ/α/ψ/H
    #  → QuadTreeFit plugs into an image-plane log-posterior for MCMC.
    # ═══════════════════════════════════════════════════════════════

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
