module EigneDemo

using Distributed
using BenchmarkTools
appproc = addprocs(4)

using LinearAlgebra, DistributedArrays, CUDA

## Build a custom type
struct DemoMatrix{T, V <: AbstractVector{T}} <: AbstractMatrix{T}
  v::V
end

Base.size(A::DemoMatrix) = length(A.v), length(A.v)
Base.getindex(A::DemoMatrix, i, j) = A.v[i] * (i == j) + A.v[i] * A.v[j]

A = DemoMatrix([1, 10, 100])
dump(A)
dump(Matrix(A))

## My very own largest eignsolver for my very own matrices
f(A::DemoMatrix) = λ -> 1 + mapreduce((v) -> v^2 / (v - λ), +, A.v)
f′(A::DemoMatrix) = λ -> mapreduce((v) -> v^2 / (v - λ)^2, +, A.v)

function LinearAlgebra.eigmax(A::DemoMatrix; tol = eps(2.0), debug = false)
  x0 = maximum(A.v) + maximum(A.v)^2
  δ = f(A)(x0) / f′(A)(x0)
  while abs(δ) > x0 * tol
    x0 -= δ
    δ = f(A)(x0) / f′(A)(x0)
    debug && println("x0 = $x0, δ = $δ") # Debugging
  end
  x0
end

eigmax(A)
eigmax(Matrix(A))

gpuA = DemoMatrix(CuArray([1, 2, 3]))
disA = DemoMatrix(distribute([1, 2, 3]))

N = 4_000_000
v = randn(N) * 0.1
A = DemoMatrix(v)
distA = DemoMatrix(distribute(v))
gpuA = DemoMatrix(CuArray(v))

# Compare Timings
@btime eigmax(A)
@btime eigmax(distA)
@btime eigmax(gpuA)
# OutofMemoryError @btime eigmax(Matrix(A))

end
