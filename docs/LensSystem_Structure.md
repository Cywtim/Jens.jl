# Jens.jl 项目结构

```
Jens.jl/
│
├── Project.toml                     # 包元数据 + 依赖
├── Manifest.toml
│
├── src/
│   ├── Jens.jl                      # 入口模块
│   │
│   ├── core/                        # ══ 核心层 ══
│   │   ├── LensUtils.jl             # 坐标变换、网格生成、椭圆率转换
│   │   ├── LensBase.jl              # AbstractLens、lens_* 接口、临界线/焦散线
│   │   ├── LensGenerator.jl         # LensedPlane、LightPlane、Grid、render_lens
│   │   └── LensSystem.jl            # ForwardModel、render(sys) 统一入口
│   │
│   ├── models/                      # ══ 模型层 ══
│   │   ├── lens/
│   │   │   ├── LensModel.jl         #   所有透镜模型的 include 汇总
│   │   │   ├── SIS.jl               #   奇异等温球 (解析 Hessian)
│   │   │   ├── SIE.jl               #   奇异等温椭球 (有限差分 Hessian)
│   │   │   ├── NIE.jl               #   非奇异等温椭球
│   │   │   ├── NIEkappa.jl          #   NIE 核 (椭圆坐标下的解析公式)
│   │   │   ├── EPL.jl               #   椭球幂律
│   │   │   ├── NFW.jl               #   Navarro-Frenk-White 暗晕
│   │   │   ├── NFWE.jl              #   NFW + 椭率
│   │   │   ├── Shear.jl             #   外部剪切 (γ₁, γ₂)
│   │   │   ├── PointMass.jl         #   点质量
│   │   │   ├── Gaussian.jl          #   高斯透镜
│   │   │   ├── SISreal.jl           #   实部 SIS (仅供测试)
│   │   │   ├── ComLens.jl           #   CombinedLens: 多模型组合 + SingleModel
│   │   │   ├── LensLOS.jl           #   ExternalTidal: LOS 潮汐矩阵
│   │   │   └── SpiralMultipole.jl   #   密度波旋臂扰动 (MGE + cos(mφ-f(R)))
│   │   │
│   │   └── light/
│   │       ├── LightModel.jl        #   所有光度模型的 include 汇总
│   │       ├── AbstractLight.jl     #   ExtendedSource、PointImage、CompositeImage
│   │       ├── SersicLight.jl       #   SersicSpheric + SersicElliptical
│   │       ├── GaussianLight.jl     #   高斯光源
│   │       ├── ExponentialLight.jl  #   指数光源
│   │       └── PointSource.jl       #   PointSource = PointImage (向后兼容别名)
│   │
│   ├── utils/                       # ══ 工具层 ══
│   │   ├── LensCosmo.jl             #   宇宙学距离比 D_ls/D_s
│   │   ├── LensPSF.jl               #   AbstractPSF + Gaussian/Moffat/Airy/Kernel PSF
│   │   ├── LensSolver.jl            #   solve_images + batch_solve_images
│   │   ├── LensNoise.jl             #   噪声模型 (Poisson/Gaussian/背景)
│   │   ├── WFC3.jl                  #   HST WFC3/UVIS 傅里叶光学 PSF
│   │   └── MGE.jl                   #   多高斯展开: fit_NFW, build_mge_lens, MGECombinedLens
│   │
│   ├── io/
│   │   └── LensFITS.jl              #   FITS 文件读写
│   │
│   ├── plotting/
│   │   ├── LensPlots.jl             #   可视化
│   │   └── LensPlot.jl              #   备用绘图 (未在include链中)
│   │
│   └── sampling/
│       └── LensTuring.jl            #   Turing.jl MCMC 接口
│
├── ext/
│   └── JensCUDA.jl                  # ══ GPU 扩展 (弱依赖 CUDA) ══
│       ├── GridGPU                  #   CuArray{Float32} 观测网格
│       ├── GenGrid_GPU              #   网格构造器
│       ├── render_lens(GridGPU)     #   GPU 光线追踪
│       └── conv_psf(CuArray)        #   cuFFT 卷积
│
├── examples/                        # ══ 示例脚本 ══
│   ├── demo_lens_system_nfw.jl      #   NFW+Shear, Sersic host + AGN, HST PSF
│   ├── demo_caustic.jl              #   临界曲线 + 焦散线
│   ├── test_batch_solver.jl         #   批量求解器对比测试
│   ├── bench_batch_solver.jl        #   求解器性能基准
│   └── demo_abstract_lens.jl        #   AbstractLens 接口示例
│
├── test/                            # ══ 测试 (Jupyter) ══
│   ├── tutorial.ipynb
│   ├── 1_test_sim_model.ipynb
│   ├── 2_test_lens_model.ipynb
│   ├── 3_test_light_model.ipynb
│   ├── 4_test_psf.ipynb
│   ├── 5_tutorial_struct.ipynb
│   ├── test_turing_lensmodel.ipynb
│   ├── test_turing_lightmodel.ipynb
│   └── test_outofdate.ipynb
│
└── docs/
    └── LensSystem_Architecture.md    #   架构文档 (类图 + 流程图)
```

## 架构分层

```
┌─────────────────────────────────────────────┐
│                  render(sys)                │  ← 用户入口
├─────────────────────────────────────────────┤
│  LensSystem   ForwardModel 统一容器          │  ← 系统层
├─────────────────────────────────────────────┤
│  LensGenerator  LensedPlane / LightPlane    │  ← 宇宙学包装
├──────────────┬──────────────────────────────┤
│  质量模型     │  光度模型                     │  ← 模型层
│  SIS SIE NIE │  ExtendedSource PointImage   │
│  NFW EPL     │  CompositeImage              │
│  Shear 组合  │  Sersic Gaussian             │
├──────────────┴──────────────────────────────┤
│  LensBase    AbstractLens / lens_* 接口     │  ← 核心抽象
├─────────────────────────────────────────────┤
│  LensUtils   ndgrid / 坐标变换 / 椭圆率      │  ← 基础工具
├─────────────────────────────────────────────┤
│  LensPSF  LensSolver  LensCosmo  MGE  WFC3  │  ← 工具层
├─────────────────────────────────────────────┤
│  ext/JensCUDA     GPU 加速 (弱依赖)          │  ← GPU 扩展
└─────────────────────────────────────────────┘
```

## 模块依赖 (include 顺序)

```
Jens.jl
  ├─(1)─ LensUtils.jl          ← 最底层，无依赖
  ├─(2)─ LensBase.jl           ← 依赖 LensUtils
  ├─(3)─ LensFITS.jl
  ├─(4)─ LensCosmo.jl
  ├─(5)─ LensLOS.jl
  ├─(6)─ LensPSF.jl
  ├─(7)─ LensSolver.jl
  ├─(8)─ LightModel.jl         ← 含 AbstractLight + Sersic + Gaussian + ...
  ├─(9)─ LensGenerator.jl      ← 依赖 Cosmo + PSF + Solver + LightModel
  ├─(10)─ LensNoise.jl
  ├─(11)─ WFC3.jl
  ├─(12)─ MGE.jl
  ├─(13)─ LensSystem.jl        ← 依赖 Generator + PSF + Solver + LightModel
  ├─(14)─ LensModel.jl         ← 所有透镜具体模型 (最后加载)
  ├─(15)─ LensPlots.jl
  └─(16)─ exports
```

## ForwardModel 数据流

```
                    质量侧                                     光度侧
            ┌──────────────────┐                    ┌──────────────────┐
  NFW  ──┐  │                  │                    │                  │
  Shear ─┼──▶ CombinedLens ────▶ LensedPlane ──┐    │  ExtendedSource ─┼──▶ CompositeImage
  SIS  ──┘  │  (线性叠加)       │  (D_ls/D_s)    │    │  PointImage    ──┘    (同红移组合)
            └──────────────────┘               │    └──────────────────┘
                                               │              │
                                               ▼              ▼
                                        ┌──────────────────────────┐
                                        │       ForwardModel       │
                                        │  lens_plane   source_plane│
                                        │  psf          grid       │
                                        └──────────────────────────┘
                                               │
                                    render(sys; solver=:batch)
                                               │
                                    ┌──────────────────────┐
                                    │     rendered image    │
                                    │  (lensed + PSF + all) │
                                    └──────────────────────┘
```