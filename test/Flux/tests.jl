using Test

using Flux, CUDA, Statistics, ProgressMeter
using Plots

@testset "test basic functions" begin
  rndmx = rand(Float32, 2, 3)
  for row in eachrow(rndmx)
    print(row)
  end

  for col in eachcol(rndmx)
    print(col)
  end

  @test xor(1, 1) == false
  @test xor(0, 0) == false
  @test xor(1, 0) == true
  @test xor(0, 1) == true

  @test size(rndmx, 1) == 2
  @test size(rndmx, 2) == 3

  Flux.glorot_uniform(3, 4)

  d = Dense(5 => 2)
  @test d(rand32(5, 64)) |> size == (2, 64)

  @test d(rand32(5, 6, 4, 64)) |> size == (2, 6, 4, 64)

  d1 = Dense(ones(2, 5), false, tanh)
  d1(ones(5))

  Flux.params(d1)
  Flux.params(d)

  xs = rand(3, 3, 3, 2)
  m = BatchNorm(3)

  Flux.trainmode!(m)
  isapprox(std(m(xs)), 1, atol = 0.1) && std(xs) != std(m(xs))

  m1 = Dense(rand(2, 3))
  @test typeof(m1.weight) == Matrix{Float64}
  m1.weight

  @test typeof(m1.bias) == Vector{Float64}
  m1.bias
  m1_gpu = m1 |> gpu
  @test typeof(m1_gpu.weight) == CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}
  @test typeof(m1_gpu.bias) == CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}
  m1_cpu = m1_gpu |> cpu
  @test typeof(m1_cpu.weight) == Matrix{Float32}
  @test typeof(m1_cpu.bias) == Vector{Float32}

  oh = Flux.onehotbatch("abracadabra", 'a':'e', 'e')
  oh1 = Flux.onehotbatch("abracadabrai", 'a':'h', 'h')
  reshape(1:15, 3, 5) * oh

  x = (a = [1, 2, 3], b = rand(6, 3))
  @test Flux.numobs(x) == 3
  x_dic = Dict(:a => [1, 2, 3], :b => rand(6, 3))
  @test Flux.numobs(x_dic) == 3
  x_dim_mis = (a = [1, 2, 3, 4], b = rand(6, 3))
  @test_throws DimensionMismatch Flux.numobs(x_dim_mis)

  @test Flux.getobs(x, 2) == (a = 2, b = x.b[:, 2])
  @test Flux.getobs(x, [1, 3]) == (a = [1, 3], b = x.b[:, [1, 3]])

  @test Flux.getobs(x_dic, 2) == Dict(:a => 2, :b => x_dic[:b][:, 2])
  @test Flux.getobs(x_dic, [1, 3]) ==
        Dict(:a => [1, 3], :b => x_dic[:b][:, [1, 3]])

  [[1, 2, 3], [4, 5, 6]] |> Flux.batch
  [1, 2, 3], [4, 5, 6] |> Flux.batch
  [(a = [1, 2], b = [3, 4]), (a = [5, 6], b = [7, 8])] |> Flux.batch
  [1 3 5 7; 2 4 6 8] |> Flux.unbatch
  [[1, 2, 3], [4, 5, 6]] |> Flux.unbatch

  Xtrain = rand(10, 100)
  array_loader = Flux.DataLoader(Xtrain, batchsize = 2)

  for x in array_loader
    @test size(x) == (10, 2)
  end

  @test array_loader.data == Xtrain

  tuple_loader = Flux.DataLoader((Xtrain,), batchsize = 2)

  for x in tuple_loader
    @test x isa Tuple{Matrix}
    @test size(x[1]) == (10, 2)
  end

  Ytrain = rand('a':'z', 100)
  train_loader = Flux.DataLoader((data = Xtrain, label = Ytrain),
    batchsize = 5,
    shuffle = true)

  for epoch in 1:100
    for (x, y) in train_loader
      @test size(x) == (10, 5)
      @test size(y) == (5,)
    end
  end

  @test first(train_loader).label isa Vector{Char}
  @test first(train_loader).label != Ytrain[1:5]

  foreach(println ∘ summary,
    Flux.DataLoader(rand(Int8, 10, 64), batchsize = 30))
end

@testset "A Neural Network in One Minute" begin
  noisy = rand(Float32, 2, 1000)
  truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]

  model = Chain(Dense(2 => 3, tanh),
    BatchNorm(3),
    Dense(3 => 2),
    softmax) |> gpu

  typeof(model)

  out1 = model(noisy |> gpu) |> cpu

  target = Flux.onehotbatch(truth, [true, false])
  data = (noisy, target)
  @test Flux.numobs(data) == 1000
  @test Flux.getobs(data, 1) == (noisy[:, 1], target[:, 1])
  @test Flux.getobs(data, 1000) == (noisy[:, 1000], target[:, 1000])

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

@testset "Fitting a Straight Line" begin
  actual(x) = 4x + 2

  x_train, x_test = hcat(0:5...), hcat(6:10...)
  y_train, y_test = x_train .|> actual, x_test .|> actual

  model = Dense(1 => 1)
  model.weight
  model.bias

  predict = Dense(1 => 1)
  x_train |> predict

  loss(model, x, y) = model(x) .- y .|> abs2 |> mean
  loss(predict, x_train, y_train)

  opt = Descent()

  data = [(x_train, y_train)]
  predict.weight
  predict.bias
  Flux.train!(loss, predict, data, opt)
  loss(predict, x_train, y_train)
  predict.weight, predict.bias

  for epoch in 1:200
    Flux.train!(loss, predict, data, opt)
  end
  loss(predict, x_train, y_train)
  predict.weight, predict.bias

  predict(x_test)
  y_test
end
