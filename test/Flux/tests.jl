using Test

using Flux, CUDA, Statistics, ProgressMeter
using Plots
using BenchmarkTools
using JLD2

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

@testset "Gradients and Layers" begin
  f(x) = 3x^2 + 2x + 1

  df(x) = gradient(f, x)[1]
  @test df(2) == 14.0

  d2f(x) = gradient(df, x)[1]
  @test d2f(2) == 6.0

  f(x, y) = (x .- y) .^ 2 |> sum
  gradient(f, [2, 1], [2, 0])

  nt = (a = [2, 1], b = [2, 0], c = tanh)
  g(x::NamedTuple) = (x.a .- x.b) .^ 2 |> sum
  @test g(nt) == 1

  dg_nt = gradient(g, nt)[1]

  gradient((x, y) -> (x.a ./ y .- x.b) .^ 2 |> sum, nt, [1, 2])

  gradient(nt, [1, 2]) do x, y
    z = x.a ./ y
    z .- x.b .^ 2 |> sum
  end

  Flux.withgradient(g, nt)

  x = [2, 1]
  y = [2, 0]
  gs = gradient(Flux.params(x, y)) do
    f(x, y)
  end
  gs[x]
  gs[y]

  W = rand(2, 5)
  b = rand(2)
  predict1(x) = W * x .+ b

  function loss(x, y)
    ŷ = predict1(x)
    (y .- ŷ) .^ 2 |> sum
  end

  x, y = rand(5), rand(2)
  loss(x, y)

  gs = gradient(() -> loss(x, y), Flux.params(W, b))
  W̄ = gs[W]
  W .-= 0.1 .* W̄
  loss(x, y)
end

@testset "Params" begin
  Flux.params(Flux.Chain(Dense(ones(2, 3)), softmax))
  bn = BatchNorm(2, relu)
  Flux.params(bn)
  Flux.params([1, 2, 3], [4])
  Flux.params([[1, 2, 3], [4]])
  Flux.params(1, [2 2], (alpha = [3, 3, 3], beta = Ref(4), gamma = sin))
end

@testset "How Flux Works" begin
  W = rand(2, 5)
  b = rand(2)
  predict1(x) = W * x .+ b

  function loss(x, y)
    ŷ = predict1(x)
    (y .- ŷ) .^ 2 |> sum
  end

  x, y = rand(5), rand(2)
  loss(x, y)

  gs = gradient(() -> loss(x, y), Flux.params(W, b))
  W̄ = gs[W]
  W .-= 0.1 .* W̄

  @test length("W̄") == 2
  @test length("W") == 1

  gradient(*, 2.0, 3.0, 5.0)
  gradient(x -> x .|> abs2 |> sum, [7.0, 11.0, 13.0])
  gradient([7, 11], 0, 1) do x, y, d
    p = size(x, d)
    x .^ p .+ y |> sum
  end
  gradient((x, y, d) -> x .^ size(x, d) .+ y |> sum, [7, 11], 0, 1)

  y, ∇ = Flux.withgradient(/, 1, 2)
  @test ∇ == gradient(/, 1, 2)
end

@testset "Building Layers" begin
  W1 = rand(3, 5)
  b1 = rand(3)
  layer1(x) = W1 * x .+ b1

  W2 = rand(2, 3)
  b2 = rand(2)
  layer2(x) = W2 * x .+ b2

  model5(x) = layer2(σ.(layer1(x)))
  model1(x) = x |> layer1 .|> σ |> layer2

  rand(5) |> model
  rand(5) |> model1

  function linear(in, out)
    W = randn(out, in)
    b = randn(out)
    x -> W * x .+ b
  end

  linear1 = linear(5, 3)
  linear2 = linear(3, 2)
  model2 = x -> x |> linear1 .|> σ |> linear2
  rand(5) |> model2

  struct Affine
    W::Any
    b::Any
  end

  Affine(in::Integer, out::Integer) = Affine(randn(out, in), randn(out))

  (m::Affine)(x) = m.W * x .+ m.b

  a = Affine(10, 5)
  a(rand(10))

  layers = [Dense(10 => 5, σ), Dense(5 => 2), softmax]
  model3(x) = foldl((x, m) -> m(x), layers, init = x)
  model3(rand(Float32, 10))

  foldl(=>, 1:4)
  foldl(=>, 1:4; init = 0)
  accumulate(=>, (1, 2, 3, 4))
  foldr(=>, 1:4)
  foldr(=>, 1:4; init = 0)

  model4 = Chain(Dense(10 => 5, σ),
    Dense(5 => 2),
    softmax)
  model4(rand(Float32, 10))

  m = softmax ∘ Dense(5 => 2) ∘ Dense(10 => 5, σ)
  m(rand(10))
  mr = x -> x |> Dense(10 => 5, σ) |> Dense(5 => 2) |> softmax
  mr(rand(10))

  m2 = Chain(x -> x^2, x -> x + 1)
  @test m2(5) == 26

  Flux.@functor Affine

  function Affine((in, out)::Pair; bias = true, init = Flux.randn32)
    W = init(out, in)
    b = Flux.create_bias(W, bias, out)
    Affine(W, b)
  end

  Affine(3 => 1, bias = false, init = ones) |> gpu
end

@testset "Training Flux Model" begin
  x = randn(28, 28)
  y = rand(10)
  data = [(x, y)]

  X = rand(28, 28, 60_000)
  size(X)
  size(eachslice(X; dims = 3))
  Y = rand(10, 60_000)
  Y |> eachcol |> size

  data2 = zip(eachslice(X; dims = 3), eachcol(Y))

  @test first(data) isa Tuple{AbstractMatrix, AbstractVector}

  data3 = Flux.DataLoader((X, Y), batchsize = 32)
  x1, y1 = first(data3)
  @test size(x1) == (28, 28, 32)
  @test size(y1) == (10, 32)
  @test length(data3) == 1875 === 60_000 ÷ 32

  y, ∇ = Flux.withgradient(/, 1, 2)
  @test Flux.withgradient(/, 1, 2).val == 0.5
  @test Flux.withgradient(/, 1, 2).grad == (0.5, -0.25)
  @test Flux.withgradient(/, 1, 2)[1] == 0.5
  @test Flux.withgradient(/, 1, 2)[2] == (0.5, -0.25)

  Flux.withgradient([1, 2, 4]) do x
    z = 1 ./ x
    sum(z), z
  end

  Flux.withgradient(3.0, 4.0) do x, y
    (div = x / y, mul = x * y)
  end

  w = [3.0]
  res = Flux.withgradient(() -> sum(abs2, w), Flux.Params([w]))

  res = Flux.withgradient(w) do x
    x .|> abs2 |> sum
  end

  a = 1
  b = 2
  a1 = (; a, b)
  @test a1.a == 1
  a2 = (a, b)
  @test_throws ErrorException a2.a

  grads = Flux.gradient(densemodel) do m
    result = m(input)
    penalty = sum(abs2, m.weight) / 2 + sum(abs2, m.bias) / 2
    my_loss(result, label) + 0.42 * penalty
  end
end

@testset "Recurrent Models" begin
  output_size = 5
  input_size = 2

  Wxh = randn(Float32, output_size, input_size)
  Whh = randn(Float32, output_size, output_size)

  b = randn(Float32, output_size)

  function rnn_cell(h, x)
    h = Wxh * x .+ Whh * h .+ b .|> tanh
    return h, h
  end

  x = rand(Float32, input_size)
  h = rand(Float32, output_size)
  h, y = rnn_cell(h, x)

  rnn = Flux.RNNCell(2, 5)
  x1 = rand(Float32, 2)
  h1 = rand(Float32, 5)
  h1, y1 = rnn(h1, x1)

  x3 = rand(Float32, 2)
  h3 = rand(Float32, 5)
  m3 = Flux.Recur(rnn, h3)

  RNN(2, 5)

  m4 = Chain(RNN(2 => 5), Dense(5 => 1))

  x4 = rand(Float32, 2)
  x4 |> m4

  x5 = [rand(Float32, 2) for _ in 1:3]
  [xi |> m4 for xi in x5]
  map(m4, x5)
  m4.(x5)
end

@testset "GPU Support" begin
  CUDA.functional()
  Flux.GPU_BACKEND

  W = cu(rand(2, 5))
  b = cu(rand(2))
  predict5(x) = W * x .+ b
  loss(x, y) = y .- predict5(x) .|> abs2 |> sum
  loss2(x, y) = sum((y .- predict5(x)) .^ 2)
  x, y = cu(rand(5)), cu(rand(2))
  @btime loss(x, y)
  @btime loss2(x, y)

  d = Dense(10 => 5, σ)
  d = fmap(cu, d)
  d.weight
  rand(10) |> cu |> d

  m6 = Chain(Dense(10 => 5, σ), Dense(5 => 2), softmax)
  m6 = fmap(cu, m6)
  rand(10) |> cu |> m6

  m7 = Dense(10, 5) |> gpu
  m8 = fmap(cu, m7)
  x7 = rand(10) |> gpu
  @btime m7(x7)

  m9 = Dense(10, 5)
  x9 = rand(10)
  @btime m9(x9)

  m100g = Dense(100, 50) |> gpu
  x100g = rand(100) |> gpu
  @btime m100g(x100g)

  m100c = Dense(100, 50)
  x100c = rand(100)
  @btime m100c(x100c)

  device = Flux.get_device(; verbose = true)
  device.deviceID

  model11 = Dense(2 => 3)
  model11.weight
  model11 = model11 |> gpu
  model11.weight
  CUDA.devices()
  device1 = Flux.get_device("CUDA", 1)
  dense_model = Dense(2 => 3)
  dense_model = dense_model |> device1
  dense_model = dense_model |> gpu
  CUDA.device(dense_model.weight)
  dense_model = dense_model |> device1

  struct MyModel
    new::Any
  end
  Flux.@functor MyModel
  model12 = MyModel(Chain(Dense(10, 5, relu), Dense(5, 2)))
  model12_state = Flux.state(model12)
  jldsave("mymodel.jld2"; model12_state)

  model12_state = JLD2.load("mymodel.jld2", "model12_state")
  # model13 = MyModel()
end
