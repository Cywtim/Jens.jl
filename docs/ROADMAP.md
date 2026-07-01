# Jens.jl 改进路线图

> 基于内部审计 + lenstronomy/PyAutoLens 等主流透镜建模包对比  
> 2026-06-26

---

## 一、内部审计发现

### ✅ 架构亮点

| 亮点 | 说明 |
|---|---|
| 类型安全的模型接口 | `LensDerivative`/`LensHessian`/`LensPotential` 函数约定 + NamedTuple 参数，编译期查错 |
| `ForwardModel` + `render()` | 统一入口，多分派自动路由到正确的渲染路径 |
| GPU 弱依赖设计 | `ext/JensCUDA.jl`，不装 CUDA 也能用 CPU |
| HMC/NUTS 支持 | `LensHMC.jl`，AdvancedHMC + FiniteDiff，绕过 ForwardDiff 限制 |
| `PointImages(intrinsic=true)` | 数据结构本身 = 同源约束，跳透镜方程 |
| 架构文档 | `docs/LensSystem_Architecture.md` — Mermaid 类图 + 流程图 + 数据流 |

### 🔴 严重问题

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| 1 | **README 空壳** | `README.md` — 9 行全是 CI badges | 新用户不知道这是什么、怎么用 |
| 2 | **0 个自动测试** | `test/runtests.jl` 是 demo 脚本不是测试 | 重构/合并零安全网 |
| 3 | **双 API 并存** | `render_lens()`(旧, LensGenerator) vs `render()`(新, LensSystem) | 用户不知道该用哪个 |
| 4 | **遗留代码未清理** | `LensInstance`, `SourceInstance`, `LensRayShooting` + Dict kwargs | 旧 API 永远不死 |

### 🟡 中等问题

| # | 问题 | 详情 |
|---|---|---|
| 5 | `src/` 里放备份 | `src/backups/LensBase_old.jl` (225行)、`LensUtils_old.jl` (262行) |
| 6 | 空文件/无用文件 | `src/plotting/LensPlot.jl` (0行)、`src/Jens_PkgTemplate.jl` (19行) |
| 7 | 重复透镜模型 | `Shear` ≡ `ShearGammaPsi` (同一物理，两种参数化)；`SIS` ≈ `SISreal` |
| 8 | `LensTuring.jl` 硬编码 | 只支持 NFW+Shear+Sersic+PointImage，不接受闭包 |
| 9 | `PointImages` 未入文档 | `docs/` 架构文档缺 PointImages、LensHMC、LensMH |
| 10 | 36 个依赖 | PyPlot + Plots + StatsPlots 三套绘图；AstroLib + LazyGrids 是否还在用？ |
| 11 | LensModel 加载顺序异常 | 透镜模型在 LensSystem 之后 include（因为 CombinedLens 依赖 ForwardModel 的 hessian dispatch） |

---

## 二、外部对比：Jens vs lenstronomy

> [lenstronomy](https://github.com/lenstronomy/lenstronomy) — Python 领域最成熟的强透镜建模包（218★，6381 commits）

| 维度 | lenstronomy | Jens |
|---|---|---|
| **模型定义** | 类继承 `LensProfileBase`，自带 `param_names`/`lower_limit`/`upper_limit` | Module + 函数约定，无自描述元数据 |
| **参数传递** | Dict → 运行时灵活但无类型检查 | NamedTuple → 编译期检查，安全 |
| **透镜组合** | `LensModel(['SIS','SHEAR'])` 字符串列表 | `CombinedLens(SIS => (...), Shear => (...))` 静态组合 |
| **MCMC** | `FittingSequence()` 一站式 API (PSO → MCMC → nested sampling) | 各自调 `lens_mh()` / `lens_hmc()` |
| **GPU** | ❌ 无原生支持 | ✅ CuArray 透传全管线 + cuFFT PSF |
| **自动微分** | JAX 分支 (herculens) 支持 | FiniteDiff 可行，ForwardDiff 被 buffer 类型阻塞 |
| **测试** | ✅ pytest 大量测试 | ❌ 0 个 `@test` |
| **文档** | ✅ readthedocs + 教程 notebooks | ⚠️ 只有架构 md，README 空 |
| **透镜方程** | 每步 solve | `PointImages` 跳过（已知像位置时） |

### 值得借鉴的 lenstronomy 设计

1. **模型自带先验范围** — 每个透镜/光度模型声明 `lower_limit`/`upper_limit`，MCMC 自动读取
2. **`FittingSequence`** — 单一入口串联 PSO → MCMC → nested sampling，用户不用手写 pipeline
3. **大量测试** — 每个模型至少测参数恢复

### Jens 独有的优势

1. **Julia 原生 JIT** — 无需 numba 预编译，改了参数直接跑
2. **类型安全** — NamedTuple 编译期检查 vs Dict 运行时炸
3. **AdvancedHMC 直接对接** — NUTS 梯度导航，窄后验也不困在局部盆地
4. **`PointImages(intrinsic=true)`** — 数据结构 = 物理约束，设计更优雅

---

## 三、render() 性能剖析

> 实测数据：128² grid, SIS + Sersic + AGN + PSF

| 步骤 | 耗时 | 占比 |
|---|---|---|
| **PSF 卷积 (FFT)** | **1.46 ms** | **65.7%** |
| Sersic 源面评估 | 0.47 ms | 21.0% |
| render_point! (4 AGN 像) | 0.16 ms | 7.2% |
| lens_derivative (偏折) | 0.14 ms | 6.2% |
| ray-trace (θ−α) | 0.01 ms | 0.6% |
| alloc + fill! | 0.005 ms | 0.2% |
| **总计** | **2.22 ms** | 100% |

**去掉 PSF: 0.66 ms (快 3.4×)**

### GPU-resident loss 加速

| 网格 | 旧 (GPU→CPU 每步) | 新 (GPU-only sum) | 加速比 |
|---|---|---|---|
| 64² | 1.03 ms | 0.39 ms | **2.64×** |
| 128² | 0.59 ms | 0.41 ms | **1.45×** |
| 256² | 1.27 ms | 0.65 ms | **1.95×** |

> 使用 `log_likelihood_gpu()` 替代手动 `Array(img)` + CPU sum，正确性 Δ = 0.0%。

---

## 四、改进任务

### P0 — 本周

- [ ] **1. 补 README.md**
  - 项目简介（一段话）
  - 5 行 quickstart（cosmology → lens → light → render → MCMC）
  - 链接到 `docs/LensSystem_Architecture.md`
  - 安装说明

- [ ] **2. 清理死代码**
  - 删除 `src/backups/LensBase_old.jl`、`src/backups/LensUtils_old.jl`
  - 删除 `src/plotting/LensPlot.jl`（0 行空文件）
  - 删除 `src/Jens_PkgTemplate.jl`（未使用模板）
  - 标注 `LensInstance`/`SourceInstance` 为 `@deprecate`

- [ ] **3. 最小测试套件**
  - `test/runtests.jl`：至少 3 个 `@test`
    - `render()` 输出尺寸正确
    - SIS 参数恢复（`lens_mh` 单参数）
    - `PointImages(intrinsic=true)` vs `PointImage` 一致性检查

- [ ] **4. 修复 LensHMC 导出**
  - `src/Jens.jl` 添加 `using .LensHMC: lens_hmc` + `export lens_hmc`

### P1 — 下个 milestone

- [ ] **5. 模型自描述元数据**
  - 每个透镜/光度模型 Module 增加：
    ```julia
    param_names = (:theta_E, :xcentre, :ycentre)
    param_lower = (0.0, -10.0, -10.0)
    param_upper = (10.0, 10.0, 10.0)
    ```
  - `lens_hmc`/`lens_mh` 可自动读取 → 用户不用手写 `lower`/`upper`

- [ ] **6. 合并/去重透镜模型**
  - `ShearGammaPsi.jl` → 合并进 `Shear.jl`（两种参数化用一个 Module）
  - `SISreal.jl` → 标注 deprecated，指向 SIS
  - 确认 `SIS` vs `NIS` (cored) 关系

- [ ] **7. LensTuring 泛化**
  - 改为接受 `(params::Vector → ForwardModel)` 闭包
  - 去掉硬编码的 NFW+Shear+Sersic+PointImage
  - 用户自己拼 ForwardModel 再传进去

- [ ] **8. 添加 `lens_fit()` 一站式 API**
  ```julia
  result = lens_fit(sys_truth, data, param_ranges;
                    sampler=:hmc, n=2000, n_warmup=500)
  # → 自动: 构建 logp → 选 sampler → 跑链 → 返回 posterior
  ```

- [ ] **9. 依赖瘦身**
  - 检查 AstroImages / LazyGrids / PyPlot 是否还在用
  - PyPlot + Plots + StatsPlots → 能否只保留一套？

- [ ] **10. 更新架构文档**
  - 补 `PointImages` 类型
  - 补 `LensHMC` / `LensMH` / `SamplingUtil`
  - 补 `LensNoise` 新 API (`GaussNoise`, `add_noise`, `log_likelihood_gpu`)

### P2 — 长期

- [ ] **11. CI/CD**
  - GitHub Actions: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - 每次 PR 自动跑测试

- [ ] **12. Documenter.jl**
  - 自动从 docstring 生成 API 文档
  - 发布到 GitHub Pages

- [ ] **13. 注册 Julia 官方包**
  - `JuliaRegistries/General` PR
  - 版本号管理 (semver)

---

## 五、近期已完成

| 日期 | 内容 |
|---|---|
| 2026-06 | `PointImages(intrinsic=true)` — 同源约束数据结构 |
| 2026-06 | `LensHMC.jl` — AdvancedHMC + FiniteDiff NUTS 采样器 |
| 2026-06 | `LensMH.jl` 改进 — 步长自适应 + `lens_mh_multistart` |
| 2026-06 | NFW GPU 兼容 — `ifelse` + 泛型类型 + 定义域钳位 |
| 2026-06 | `LensNoise.jl` 重构 — `GaussNoise`/`PoissNoise` 类型 + `add_noise` + `log_likelihood_gpu` |
| 2026-06 | `gpu_grid()` 便捷函数 |
| 2026-06 | render 性能剖析 — PSF 卷积 = 66%，GPU-resident loss = 1.5-2.6× 加速 |