module LensTuring

    #  LensTuring — Turing.jl MCMC bridge for Jens.jl ForwardModel
    #
    #  Usage:
    #    model = lens_fit_model(data, grid, psf, cosmo, z_lens, z_src)
    #    chain = sample(model, NUTS(), 500)
    #
    #  The generated @model function creates a fresh ForwardModel
    #  at each MCMC iteration from the proposed parameters, renders
    #  via `render(sys; solver=:batch)`, and computes Gaussian
    #  log-likelihood.

    using Turing, Distributions, LinearAlgebra
    using Jens
    using Jens.LensModel.ComLens: CombinedLens
    using Jens.LensModel: NFW, Shear, SIS
    using Jens.LightModel: ExtendedSource, PointImage, CompositeImage
    using Jens.LightModel.SersicLight: SersicSpheric
    using Jens.LensGenerator: LensedPlane, LightPlane
    using Jens.LensSystem: ForwardModel, render
    using Cosmology

    export lens_fit_model, lens_fit_model_simple

    # ══════════════════════════════════════════════════════════════
    #  Full model: NFW + Shear lens, Sersic host + AGN point source
    # ══════════════════════════════════════════════════════════════

    """
        model = lens_fit_model(data, grid, psf, cosmo, z_lens, z_src)

    Build a Turing `@model` for fitting NFW+Shear lens + Sersic+AGN light.

    # Free parameters (8)
      Lens:  Rs, alpha_Rs, gamma1, gamma2
      Light: amp_host, Rsersic_host, n_host, flux_agn

    # Fixed
      Lens centre = (0,0), host centre = (0,0), AGN position = (0.05, -0.03)
      sigma_noise = median(abs.(data)) * 0.02   (2% of median as noise floor)
    """
    function lens_fit_model(data, grid, psf, cosmo, z_lens, z_src)
        sigma_noise = max(median(abs.(data)) * 0.02, 1e-6)

        @model function _lens_fit(data)
            # ── Lens mass priors ──────────────────────
            Rs       ~ Uniform(1.0, 10.0)
            alpha_Rs ~ Uniform(0.1, 3.0)
            gamma1   ~ Normal(0.0, 0.1)
            gamma2   ~ Normal(0.0, 0.1)

            # ── Host light priors ─────────────────────
            amp_host     ~ Uniform(0.1, 5.0)
            Rsersic_host ~ Uniform(0.05, 1.0)
            n_host       ~ Uniform(0.5, 6.0)

            # ── AGN flux prior ────────────────────────
            flux_agn ~ Uniform(10.0, 500.0)

            # ── Build forward model ────────────────────
            lens_mass = CombinedLens(
                NFW   => (Rs=Rs, alpha_Rs=alpha_Rs,
                          xcentre=0.0, ycentre=0.0),
                Shear => (gamma1=gamma1, gamma2=gamma2,
                          xcentre=0.0, ycentre=0.0),
            )
            lp = LensedPlane(lens_mass; z_lens=z_lens, cosmology=cosmo)

            host = ExtendedSource(SersicSpheric;
                amp     = amp_host,
                Rsersic = Rsersic_host,
                n       = n_host,
                xcentre = 0.0,
                ycentre = 0.0,
            )
            agn = PointImage(flux=flux_agn,
                             beta_x=0.05, beta_y=-0.03)
            src = CompositeImage(host, agn)

            sys = ForwardModel(
                lens_plane   = lp,
                source_plane = LightPlane(src; z=z_src),
                grid         = grid,
                psf          = psf,
            )

            # ── Forward render ─────────────────────────
            img = render(sys; solver=:batch)

            # ── Likelihood ─────────────────────────────
            for i in eachindex(data)
                data[i] ~ Normal(img[i], sigma_noise)
            end
        end

        return _lens_fit(data)
    end


    # ══════════════════════════════════════════════════════════════
    #  Simple model: SIS lens, Sersic host (no AGN) — fast debug
    # ══════════════════════════════════════════════════════════════

    """
        model = lens_fit_model_simple(data, grid, psf, cosmo, z_lens, z_src)

    Minimal 3-parameter model for quick testing:
      SIS theta_E + Sersic amp, Rsersic.
    """
    function lens_fit_model_simple(data, grid, psf, cosmo, z_lens, z_src)
        sigma_noise = max(median(abs.(data)) * 0.02, 1e-6)

        @model function _simple_fit(data)
            theta_E ~ Uniform(0.1, 2.0)
            amp     ~ Uniform(0.1, 5.0)
            Rsersic ~ Uniform(0.05, 1.0)

            lens_mass = CombinedLens(
                SIS => (theta_E=theta_E, xcentre=0.0, ycentre=0.0),
            )
            lp = LensedPlane(lens_mass; z_lens=z_lens, cosmology=cosmo)

            host = ExtendedSource(SersicSpheric;
                amp=amp, Rsersic=Rsersic, n=2.0,
                xcentre=0.0, ycentre=0.0)

            sys = ForwardModel(
                lens_plane   = lp,
                source_plane = LightPlane(host; z=z_src),
                grid         = grid,
                psf          = psf,
            )

            img = render(sys; solver=:batch)

            for i in eachindex(data)
                data[i] ~ Normal(img[i], sigma_noise)
            end
        end

        return _simple_fit(data)
    end

end