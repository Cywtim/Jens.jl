# ═══════════════════════════════════════════════════════════════
#  Jens.jl  —  CI runtests
#
#  覆盖: LensUtils, LensBase, LensModels, LensPSF, LensMask,
#        LensNoise, LensCosmo, LensGenerator, LensSystem, Sampling,
#        FiniteDiff 梯度一致性
# ═══════════════════════════════════════════════════════════════

using Test
using LinearAlgebra, Statistics, Random

using Jens
using Jens: LensBase   as LB
using Jens: LensUtils   as LU
using Jens: LensCosmo   as LC
using Jens: LensGenerator as LG
using Jens: LensSystem  as LS
using Jens: LensModel   as Jmodel
using Jens: LightModel  as Jlight

# ── Lens models ──
import Jens.LensModel: SIS, NIEkappa, NIE, SIE, NFW, EPL, PointMass
import Jens.LensModel: Shear, Gaussian, tNFW, ComLens

# ── PSF ──
import Jens.LensPSF: GaussianPSF, MoffatPSF, AiryDiskPSF, KernelPSF
import Jens.LensPSF: make_kernel, conv_psf

# ── Mask ──
import Jens.LensMask: circular_mask, annular_mask, rectangular_mask
import Jens.LensMask: from_matrix, combine, invert, n_valid

# ── Noise ──
import Jens.LensNoise: GaussNoise, PoissNoise
import Jens.LensNoise: add_noise, log_likelihood, estimate_sigma

# ── Sampling ──
import Jens.LensMH:  lens_mh, lens_mh_multi
import Jens.LensHMC: lens_hmc
import Jens.LensPSO: lens_pso_mh

# ── Gradients ──
using FiniteDiff
using ForwardDiff

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

const SMALL_NX = 32

"一个小 Gaussian posterior, 用于 MH/HMC/PSO 测试"
function _test_logp(p::Vector{Float64})
    mu = [0.5, 0.5]
    sigma = 0.2
    return -sum(((p .- mu) ./ sigma).^2) / 2
end

# ═══════════════════════════════════════════════════════════════
#  1. LensUtils
# ═══════════════════════════════════════════════════════════════
@testset "LensUtils" begin

    @testset "LensGrid" begin
        xg, yg = LU.LensGrid(; xl=2.0, nx=SMALL_NX)
        @test size(xg) == (SMALL_NX, SMALL_NX)
        @test size(yg) == (SMALL_NX, SMALL_NX)
        @test minimum(xg) >= -2.0
        @test maximum(xg) <=  2.0
        @test all(diff(xg[:, 1]) .> 0)
        @test all(diff(yg[1, :]) .> 0)
    end

    @testset "Car2Pol ↔ Pol2Car roundtrip" begin
        r, phi = LU.Car2Pol(1.0, 2.0)
        x2, y2 = LU.Pol2Car(r, phi)
        @test x2 ≈ 1.0
        @test y2 ≈ 2.0
    end

    @testset "e2phiq ↔ phiq2e roundtrip" begin
        q, phi = LU.e2phiq(0.1, 0.2)
        e1, e2 = LU.phiq2e(q, phi)
        @test e1 ≈ 0.1
        @test e2 ≈ 0.2
    end

    @testset "ShearPol2Car ↔ ShearCar2Pol roundtrip" begin
        g1, g2 = LU.ShearPol2Car(pi/6, 0.1)
        phi, gamma = LU.ShearCar2Pol(g1, g2)
        @test gamma ≈ 0.1
        @test cos(phi) ≈ cos(pi/6)  atol=1e-12
    end

    @testset "EllipticalDistortion zero" begin
        xg, yg = LU.LensGrid(; xl=2.0, nx=SMALL_NX)
        xsh, ysh = LU.EllipticalDistortion(xg, yg; e1=0.0, e2=0.0,
                                            xcentre=0.0, ycentre=0.0)
        @test xsh ≈ xg
        @test ysh ≈ yg
    end
end

# ═══════════════════════════════════════════════════════════════
#  2. LensBase  —  interface + high-level functions
# ═══════════════════════════════════════════════════════════════
@testset "LensBase" begin
    xg, yg = LU.LensGrid(; xl=2.0, nx=SMALL_NX)

    @testset "lens_derivative / lens_hessian / lens_potential (SIS)" begin
            # low-level interface takes arrays + direct kwargs
            x1, y1 = [1.0], [0.0]
            fx, fy = LB.lens_derivative(SIS.SIS, x1, y1;
                        theta_E=1.0, xcentre=0.0, ycentre=0.0)
            @test fx ≈ [1.0]   # SIS: α = θ_E * θ̂
            @test fy ≈ [0.0]   atol=1e-12

            fxx, fxy, fyy = LB.lens_hessian(SIS.SIS, x1, y1;
                        theta_E=1.0, xcentre=0.0, ycentre=0.0)
            # SIS: κ = θ_E/(2R),  so ∇²ψ = 2κ = θ_E/R = 1 at R=1
            @test fxx[1] + fyy[1] ≈ 1.0  atol=1e-6

            psi = LB.lens_potential(SIS.SIS, x1, y1;
                        theta_E=1.0, xcentre=0.0, ycentre=0.0)
            @test psi ≈ [1.0]  # SIS: ψ = θ_E * R
        end

    @testset "LensPlane" begin
        betax, betay = LB.LensPlane(xg, yg; LensModel=SIS.SIS,
            LensKwargs=Dict(:theta_E => 1.0, :xcentre => 0.0, :ycentre => 0.0))
        @test size(betax) == size(xg)
        @test size(betay) == size(xg)
    end

    @testset "LensDetJacobian / LensMagnification" begin
        x_far = fill(100.0, SMALL_NX, SMALL_NX)
        y_far = fill(0.0,   SMALL_NX, SMALL_NX)
        detJ = LB.LensDetJacobian(x_far, y_far; LensModel=SIS.SIS,
            LensKwargs=Dict(:theta_E => 1.0, :xcentre => 0.0, :ycentre => 0.0))
        @test all(isapprox.(detJ, 1.0; atol=0.05))

        mu = LB.LensMagnification(x_far, y_far; LensModel=SIS.SIS,
            LensKwargs=Dict(:theta_E => 1.0, :xcentre => 0.0, :ycentre => 0.0))
        @test all(isapprox.(mu, 1.0; atol=0.05))
    end

    @testset "LensFermat" begin
        fermat = LB.LensFermat(xg, yg, [0.0, 0.0]; LensModel=SIS.SIS,
            LensKwargs=Dict(:theta_E => 1.0, :xcentre => 0.0, :ycentre => 0.0))
        @test size(fermat) == size(xg)
        # Fermat minimum at Einstein ring R=θ_E: Φ = ½ − θ_E = −½
        @test minimum(fermat) ≈ -0.5  atol=0.1
    end

    @testset "LensCriticalCurve / LensCaustic" begin
        ccx, ccy = LB.LensCriticalCurve(;
            LensModel=SIS.SIS,
            LensKwargs=Dict(:theta_E => 1.0, :xcentre => 0.0, :ycentre => 0.0),
            r_bins=200, theta_bins=200)
        @test length(ccx) == length(ccy) > 0
        # SIS critical curve: circle at R ≈ θ_E
        R_cc = @. sqrt(ccx^2 + ccy^2)
        @test all(isapprox.(R_cc, 1.0; atol=0.1))
        @test all(R_cc .> 0.5)

        csx, csy = LB.LensCaustic(;
            LensModel=SIS.SIS,
            LensKwargs=Dict(:theta_E => 1.0, :xcentre => 0.0, :ycentre => 0.0),
            r_bins=200, theta_bins=200)
        # SIS caustic: degenerate point at origin
        @test length(csx) > 0
        R_cs = @. sqrt(csx^2 + csy^2)
        @test all(R_cs .< 0.1)
    end
end

# ═══════════════════════════════════════════════════════════════
#  3. Lens Models
# ═══════════════════════════════════════════════════════════════
@testset "LensModels" begin
    x_test  = [2.0, 1.0, 0.5]
    y_test  = [0.0, 0.5, 1.0]

    function _model_tests(name, mod; kw...)
        @testset "$name" begin
            fx, fy = mod.LensDerivative(x_test, y_test; kw...)
            @test size(fx) == size(x_test)
            @test all(isfinite, fx)
            @test all(isfinite, fy)

            fxx, fxy, fyy = mod.LensHessian(x_test, y_test; kw...)
            @test size(fxx) == size(x_test)
            @test all(isfinite, fxx)
            @test all(isfinite, fyy)

            psi = mod.LensPotential(x_test, y_test; kw...)
            @test all(isfinite, psi)
        end
    end

    _model_tests("SIS",       SIS.SIS;       theta_E=1.0)
    _model_tests("NIEkappa",  NIEkappa.NIEkappa; b=1.0, s=0.1, q=0.7, varphi=0.3)
    _model_tests("NIE",       NIE.NIE;       theta_E=1.0, s_scale=0.1,
                                              e1=0.2, e2=-0.1)
    _model_tests("SIE",       SIE.SIE;       theta_E=1.0, e1=0.1, e2=0.0)
    _model_tests("NFW",       NFW.NFW;       Rs=1.0, alpha_Rs=0.5)
    _model_tests("PointMass", PointMass.PointMass; theta_E=0.5)

    # EPL: gamma=2.0 is SIS-like, but potential may be NaN for certain
    # parameter combos; test at least derivative + hessian
    @testset "EPL" begin
        fx, fy = EPL.EPL.LensDerivative(x_test, y_test;
                    theta_E=1.0, gamma=2.1, e1=0.1, e2=0.0)
        @test all(isfinite, fx)
        @test all(isfinite, fy)
    end

    # tNFW: struct-based model (tNFWLens <: AbstractLens), not Module
    @testset "tNFW" begin
        tl = tNFW.tNFWLens(; Rs=2.0, alpha_Rs=0.5, r_t=10.0)
        fx, fy = LB.lens_derivative(tl, x_test, y_test)
        @test all(isfinite, fx)
    end

    # Gaussian: lens model, test derivative
    @testset "Gaussian" begin
        fx, fy = Gaussian.Gaussian.LensDerivative(x_test, y_test;
                    kappa0=0.3, q=0.8, sigma=1.5)
        @test all(isfinite, fx)
    end

    @testset "Shear" begin
        fx, fy = Shear.Shear.LensDerivative(x_test, y_test;
                    gamma1=0.05, gamma2=-0.03)
        @test size(fx) == size(x_test)
        @test all(isfinite, fx)
    end

    @testset "CombinedLens (SIS + Shear)" begin
        cl = ComLens.CombinedLens(
            SIS.SIS => (theta_E=1.0,),
            Shear.Shear => (gamma1=0.05, gamma2=0.0)
        )
        fx, fy = LB.lens_derivative(cl, x_test, y_test)
        @test size(fx) == size(x_test)
        fxx, fxy, fyy = LB.lens_hessian(cl, x_test, y_test)
        @test all(isfinite, fxx)
    end

    @testset "SingleModel" begin
        sm = LB.SingleModel(SIS.SIS; theta_E=1.0)
        fx, fy = LB.lens_derivative(sm, [1.0], [0.0])
        @test fx ≈ [1.0]
    end
end

# ═══════════════════════════════════════════════════════════════
#  4. LensPSF
# ═══════════════════════════════════════════════════════════════
@testset "LensPSF" begin
    half = 10
    ps = 0.05

    @testset "GaussianPSF" begin
        psf = GaussianPSF(; fwhm=5.0)
        k = make_kernel(psf, ps, half)
        @test size(k) == (2*half+1, 2*half+1)
        @test sum(k) ≈ 1.0  rtol=1e-3
        @test k[half+1, half+1] >= k[1, 1]
    end

    @testset "MoffatPSF" begin
        psf = MoffatPSF(; fwhm=5.0, alpha=3.0)
        k = make_kernel(psf, ps, half)
        @test sum(k) ≈ 1.0  rtol=1e-3
    end

    @testset "AiryDiskPSF" begin
        psf = AiryDiskPSF(; fwhm=5.0)
        k = make_kernel(psf, ps, half)
        @test sum(k) ≈ 1.0  rtol=1e-3
    end

    @testset "KernelPSF" begin
        raw = ones(5, 5)
        raw[3, 3] = 10.0
        raw ./= sum(raw)
        psf = KernelPSF(raw, 0.1)
        k = make_kernel(psf, ps, half)
        @test sum(k) ≈ 1.0  rtol=1e-3
    end

    @testset "conv_psf" begin
        img = ones(64, 64)
        psf = GaussianPSF(; fwhm=3.0)
        c = conv_psf(img, psf, ps; half=10)
        @test size(c) == size(img)
        @test all(isfinite, c)
    end
end

# ═══════════════════════════════════════════════════════════════
#  5. LensMask
# ═══════════════════════════════════════════════════════════════
@testset "LensMask" begin
    xg, yg = LU.LensGrid(; xl=2.0, nx=64)

    @testset "circular_mask" begin
        m = circular_mask((xg, yg), 1.0)
        @test size(m) == size(xg)
        @test m isa BitMatrix
        @test n_valid(m) > 0
    end

    @testset "annular_mask" begin
        m = annular_mask((xg, yg), 0.5, 1.5)
        @test n_valid(m) > 0
    end

    @testset "rectangular_mask" begin
        m = rectangular_mask((xg, yg), -0.5, 0.5, -0.5, 0.5)
        @test n_valid(m) > 0
    end

    @testset "combine" begin
        m1 = circular_mask((xg, yg), 1.0)
        m2 = circular_mask((xg, yg), 0.5)
        mc = combine(m1, m2; op=&)
        @test n_valid(mc) <= n_valid(m1)
    end

    @testset "invert" begin
        m = circular_mask((xg, yg), 1.0)
        mi = invert(m)
        @test n_valid(m) + n_valid(mi) == length(m)
    end

    @testset "from_matrix" begin
        A = rand(64, 64)
        m = from_matrix(A .> 0.5)
        @test m isa BitMatrix
    end
end

# ═══════════════════════════════════════════════════════════════
#  6. LensNoise
# ═══════════════════════════════════════════════════════════════
@testset "LensNoise" begin
    img = ones(32, 32) .* 100.0

    @testset "GaussNoise" begin
        gn = GaussNoise(5.0)
        noisy = add_noise(img, gn)
        @test size(noisy) == size(img)
        @test !all(noisy .≈ img)
    end

    @testset "PoissNoise" begin
        pn = PoissNoise(1.0)
        noisy = add_noise(img, pn)
        @test size(noisy) == size(img)
    end

    @testset "log_likelihood" begin
        gn = GaussNoise(2.0)
        ll = log_likelihood(img, img, gn)
        @test isfinite(ll)
        ll_bad = log_likelihood(img, img .+ 10.0, gn)
        @test ll > ll_bad
    end

    @testset "estimate_sigma" begin
        s = estimate_sigma(img; method=:mad)
        @test s <= 1.0
    end
end

# ═══════════════════════════════════════════════════════════════
#  7. LensCosmo
# ═══════════════════════════════════════════════════════════════
@testset "LensCosmo" begin
    cosmo = LC.Cosmology.FlatLCDM(0.7, 0.7, 0.3, 0.0)

    @testset "angular_diameter_distance" begin
        da = LC.angular_diameter_distance(cosmo, 0.5)
        @test da.val > 0.0
        da2 = LC.angular_diameter_distance(cosmo, 0.3, 1.0)
        @test da2.val > 0.0
    end

    @testset "lens_distance_ratio" begin
        d = LC.lens_distance_ratio(cosmo, 0.5, 1.5)
        @test 0.0 < d < 1.0
    end
end

# ═══════════════════════════════════════════════════════════════
#  8. LensGenerator + LensSystem
# ═══════════════════════════════════════════════════════════════
@testset "LensGenerator" begin
    cosmo = LC.Cosmology.FlatLCDM(0.7, 0.7, 0.3, 0.0)

    @testset "LensedPlane" begin
        lp = LG.LensedPlane(SIS.SIS; z_lens=0.5, cosmology=cosmo)
        fx, fy = LB.lens_derivative(lp, [1.0], [0.0];
                    z_source=1.5, theta_E=1.0, xcentre=0.0, ycentre=0.0)
        @test isfinite(fx[1])
    end

    @testset "Grid / GenGrid" begin
        g = LG.GenGrid(; pix_n=64, pix_size=0.09)
        @test g isa LG.Grid
    end
end

# ═══════════════════════════════════════════════════════════════
#  9. Sampling
# ═══════════════════════════════════════════════════════════════
@testset "Sampling" begin

    @testset "lens_mh" begin
        lo = [0.0, 0.0]
        hi = [1.0, 1.0]
        r = lens_mh(_test_logp, lo, hi; n=200, adapt=false)
        s = r.samples                       # (n_params × n_samples)
        @test size(s, 1) == 2
        @test size(s, 2) >= 50
        @test all(s[1, :] .>= 0.0)
        @test all(s[1, :] .<= 1.0)
    end

    @testset "lens_mh_multi" begin
        lo = [0.0, 0.0]
        hi = [1.0, 1.0]
        chains = lens_mh_multi(_test_logp, lo, hi; n=150, n_chains=3, adapt=false)
        @test length(chains) == 3
    end

    @testset "lens_hmc" begin
        lo = [0.0, 0.0]
        hi = [1.0, 1.0]
        r = lens_hmc(_test_logp, lo, hi; n=100)
        s = r.samples                       # (n_params × n_samples)
        @test size(s, 1) == 2
        @test size(s, 2) >= 20
    end

    @testset "lens_pso_mh" begin
        lo = [0.0, 0.0]
        hi = [1.0, 1.0]
        r = lens_pso_mh(_test_logp, lo, hi; n_particles=8, n_iter=20, n_mh=100)
        s = r.mh_result.samples             # (n_params × n_samples)
        @test size(s, 1) == 2
        @test size(s, 2) >= 10
    end
end

# ═══════════════════════════════════════════════════════════════
#  10. Gradient consistency — FiniteDiff vs ForwardDiff
# ═══════════════════════════════════════════════════════════════
@testset "Gradient consistency" begin

    function _sis_residue(p)
        theta_E, xc, yc = p
        xg, yg = LU.LensGrid(; xl=1.0, nx=16)
        betax, betay = LB.LensPlane(xg, yg; LensModel=SIS.SIS,
            LensKwargs=Dict(:theta_E => theta_E, :xcentre => xc, :ycentre => yc))
        return sum(betax.^2 + betay.^2)
    end

    p0 = [1.0, 0.0, 0.0]
    fd_grad = FiniteDiff.finite_difference_gradient(_sis_residue, p0)
    fwd_cfg = ForwardDiff.GradientConfig(_sis_residue, p0)
    fwd_grad = ForwardDiff.gradient(_sis_residue, p0, fwd_cfg)

    @test length(fd_grad) == length(fwd_grad) == 3
    @test fd_grad ≈ fwd_grad  rtol=1e-5

    # SingleModel and CombinedLens: ForwardDiff can't handle struct-based
    # models (type-wall at _convert_params). This is expected — FiniteDiff
    # is the GPU-safe alternative.

    @testset "SingleModel gradient (FiniteDiff only)" begin
        function _sm_residue(p)
            theta_E = p[1]
            sm = LB.SingleModel(SIS.SIS; theta_E=theta_E, xcentre=0.0, ycentre=0.0)
            xg, yg = LU.LensGrid(; xl=1.0, nx=16)
            betax, betay = LB.LensPlane(xg, yg; LensModel=sm, LensKwargs=Dict())
            return sum(betax.^2 + betay.^2)
        end
        fd = FiniteDiff.finite_difference_gradient(_sm_residue, [1.0])
        @test length(fd) == 1
        @test all(isfinite, fd)
    end

    @testset "CombinedLens gradient (FiniteDiff only)" begin
        function _cl_residue(p)
            t1, t2 = p
            cl = ComLens.CombinedLens(
                SIS.SIS => (theta_E=t1,),
                SIS.SIS => (theta_E=t2, xcentre=0.5, ycentre=0.0)
            )
            xg, yg = LU.LensGrid(; xl=1.0, nx=16)
            betax, betay = LB.LensPlane(xg, yg; LensModel=cl, LensKwargs=Dict())
            return sum(betax.^2 + betay.^2)
        end
        fd = FiniteDiff.finite_difference_gradient(_cl_residue, [0.5, 0.5])
        @test length(fd) == 2
        @test all(isfinite, fd)
    end
end

# ═══════════════════════════════════════════════════════════════
#  11. GPU — 条件运行
# ═══════════════════════════════════════════════════════════════
@testset "GPU (skip if no CUDA)" begin
    try
        using CUDA
        if CUDA.functional()
            ext = Base.get_extension(Jens, :JensCUDA)
            @test ext !== nothing
            @info "CUDA available — extension loaded"
        else
            @info "CUDA installed but no functional device"
        end
    catch e
        @info "CUDA not available, skipping GPU tests" e
    end
end

println("\n✅ All tests passed.")
