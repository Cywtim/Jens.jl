using Jens

# including packages
using LazyGrids, PyPlot, BenchmarkTools
using PSFModels, ImageFiltering

using Jens
using Jens:LensBase as LB
using Jens:LensUtils as LU
using Jens:LensFITS as LF
using Jens:LensGenerator as LG
using Jens:LensModel as Jmodel
using Jens:LightModel as Jlight

using Jens.LensModel:NIE as NIE
using Jens.LightModel:GaussianLight as GaussianLight

xg, yg = LU.LensGrid(;xl=2., nx=51)

# The Lensed Plane of Grids
LensParaDict = Dict(:b => 1., :s => 0.1,
 :q => 0.7, :varphi =>  - pi/6. ,
  :xcentre => 0.1, :ycentre => .0);
betax, betay = LB.LensPlane(xg, yg; LensModel=NIE.NIEkappa, LensKwargs=LensParaDict)

FigLnes = PyPlot.figure(figsize=(6, 6))
PyPlot.scatter(betax, betay, s=1)
PyPlot.xlabel("x (pix)")
PyPlot.ylabel("y (pix)")

print("finished")