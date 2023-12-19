using JuliaML
using Test

tests = [
  "Flux",
]
if !isempty(ARGS)
  tests = ARGS  # Set list to same as command line args
end

@testset "All Tests" begin
  for t in tests
    include("$(t)/tests.jl")
  end
end
