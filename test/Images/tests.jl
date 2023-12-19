using Test
using Images, TestImages, Colors, ImageInTerminal
using FixedPointNumbers
using Plots

function add_marker(image::AbstractArray, x::Int, y::Int)
  marked_image = copy(image)
  mark_color = RGB(1, 0, 0)  # Red color for the marker    
  # Draw a cross-shaped marker at the specified position
  for i in -5:5
    marked_image[y + i, x] = mark_color
    marked_image[y, x + i] = mark_color
  end
  return marked_image
end

function add_line(image::AbstractArray, x1::Int, y1::Int, x2::Int, y2::Int)
  marked_image = copy(image)
  line_color = RGB(0, 1, 0)  # Green color for the line    
  # Calculate the slope and intercept of the line
  slope = (y2 - y1) / (x2 - x1)
  intercept = y1 - slope * x1
  # Draw the line by setting the corresponding pixels
  for x in min(x1, x2):max(x1, x2)
    y = round(slope * x + intercept) |> Int
    marked_image[y, x] = line_color
  end
  return marked_image
end

@testset "Add marker in image" begin
  img = testimage("lighthouse")
  size(img)
  summary(img)

  # Define the key points
  keypoints = [(100, 200), (300, 400), (500, 600)]
  # Plot the image
  plot(img)
  # Add key points to the plot
  scatter!([kp for kp in keypoints],
    markersize = 3,
    markercolor = :blue)

  # Add a marker at position (100, 200)
  marked_image = add_marker(img, 100, 200)
  marked_image = add_marker(marked_image, 300, 400)
  marked_image = add_marker(marked_image, 450, 450)
  # Display the original and marked images
  display(img)
  display(marked_img)

  marked_line_image = add_line(img, 100, 200, 300, 400)
end

@testset "Basic images" begin
  methods(testimage)
  img = testimage("lighthouse")
  size(img)
  summary(img)
  img[1]
  @which []
  c = img[10]
  println(summary(img))
  println(c.r, " ", c.g, " ", c.b)
  println(red(c), " ", green(c), " ", blue(c))

  img_bgr = BGR.(img)
  println(summary(img_bgr))

  c2 = img_bgr[10]
  println(c2.r, " ", c2.g, " ", c2.b)
  println(red(c2), " ", green(c2), " ", blue(c2))
  img2 = load("test/Images/Guggenheim_Museum_Bilbao.JPG")

  tiled_img = Array{RGB{N0f8}, 2}(undef, size(img))
  LinRange(1, size(img, 1), 8)
  LinRange(1, size(img, 1), 8) .|> Int16
  img[1:8:size(img, 1)]
  stepsize = 16
  img[1:stepsize:size(img, 1), 1:stepsize:size(img, 2)]
  maximum(green.(img[1:10]))
  maximum(green.(img[1:16, 1:16]))

  tiled_img2 = similar(img)
  summary(tiled_img2)

  # for x in 1:stepsize:size(img, 1)
  #   for y in 1:stepsize:size(img, 2)
  #     x_end = min(x + stepsize, size(img, 1))
  #     y_end = min(y + stepsize, size(img, 2))
  #     imgv = @view img[x:x_end, y:y_end]
  #     tiled_img[x:x_end, y:y_end] = RGB{N0f8}(maximum(red.(imgv)),
  #       maximum(green.(imgv)), maximum(blue.(imgv)))
  #   end
  # end
end

@testset "From JuliaImages" begin
  img3 = rand(2, 2)
  a = [1, 2, 3, 4]
  map(Float64, a)
  Float64.(a)
  img3g = Gray.(img3)
  display(MIME("text/plain"), img3g)
  dump(img3g[1, 1])
  sizeof(img3)
  sizeof(img3g)
  @test img3 == img3g

  imgc = rand(RGB{Float32}, 2, 2)
  println(imgc)
  size(imgc)
  dump(imgc[1, 1])
  dump(imgc[4])

  c = imgc[1, 1]
  (red(c), green(c), blue(c))
  dump(BGR(c))
  c24 = RGB24(c)
  dump(c24)
  c24.color
  0x3f / 0xff
  0xca / 0xff
  0xdd / 0xff
  r = red(c24)
  dump(r)
  # r == 0x3f/0xff

  csat = RGB{Float32}(8, 2, 0)
  [csat / 2^i for i in 0:7]

  @test_throws ArgumentError RGB(8, 2, 0)
  RGB(1, 0, 0)

  (typemax(N0f8), eps(N0f8))
  (typemax(N0f16), eps(N0f16))
  [Gray(N0f16(0)) Gray(N0f16(1)) Gray(reinterpret(N0f16, 0x0fff)) Gray(N0f16(0))]
  (typemax(N4f12), eps(N4f12))
  [Gray(N4f12(0)) Gray(N4f12(1)) Gray(reinterpret(N4f12, 0x0fff)) Gray(N4f12(0))]
  0xff + 0xff
  1N0f8 + 1N0f8
  0xfe / 0xff
end
