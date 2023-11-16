using Test

using Flux, CUDA, Statistics, ProgressMeter
using Plots

@testset "A Neural Network in One Minute" begin
  noisy = rand(Float32, 2, 1000)
  truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]

  model = Chain(Dense(2 => 3, tanh),
    BatchNorm(3),
    Dense(3 => 2),
    softmax) |> gpu

  out1 = model(noisy |> gpu) |> cpu

  target = Flux.onehotbatch(truth, [true, false])
  loader = Flux.DataLoader((noisy, target) |> gpu,
    batchsize = 64,
    shuffle = true)

  optim = Flux.setup(Flux.Adam(0.01), model)

  losses = []

  @showprogress for each in 1:1_000
    for (x, y) in loader
      loss, grads = Flux.withgradient(model) do m
        y_hat = m(x)
        Flux.crossentropy(y_hat, y)
      end
      Flux.update!(optim, model, grads[1])
      push!(losses, loss)
    end
  end

  optim

  out2 = model(noisy |> gpu) |> cpu

  mean((out2[1, :] .> 0.5) .== truth)

  p_true = scatter(noisy[1, :],
    noisy[2, :],
    zcolor = truth,
    title = "True classification",
    legend = false)

  p_raw = scatter(noisy[1, :],
    noisy[2, :],
    zcolor = out1[1, :],
    title = "Untrained network",
    label = "",
    clims = (0, 1))

  p_done = scatter(noisy[1, :],
    noisy[2, :],
    zcolor = out2[1, :],
    title = "Trained network",
    legend = false)

  plot(p_true, p_raw, p_done, layout = (1, 3), size = (1000, 330))

  plot(losses; xaxis = (:log10, "iteration"),
    yaxis = "loss", label = "per batch")

  n = length(loader)
  plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)),
    label = "epoch mean", dpi = 200)
end
