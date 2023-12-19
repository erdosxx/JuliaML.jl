function T(𝛉::AbstractArray,
    𝒞::Tuple{AbstractArray,
      Vararg{AbstractArray}},
    𝒟::Tuple{AbstractArray, Vararg{AbstractArray}})
  ⊗ = kron
  l = length(𝛉)
  𝐈ₗ = SMatrix{l, l}(1.0I)
  𝐈ₘ = SMatrix{1, 1}(1.0I)
  𝐓 = @SMatrix zeros(l, l)
  N = length(𝒟[1])
  ℳ, ℳʹ = 𝒟
  Λ₁, Λ₂ = 𝒞
  𝚲ₙ = @MMatrix zeros(4, 4)
  𝐞₁ = @SMatrix [1.0; 0.0; 0.0]
  𝐞₂ = @SMatrix [0.0; 1.0; 0.0]
  for n in 1:N
    index = SVector(1, 2)
    𝚲ₙ[1:2, 1:2] .= Λ₁[n][index, index]
    𝚲ₙ[3:4, 3:4] .= Λ₂[n][index, index]
    𝐦 = hom(ℳ[n])
    𝐦ʹ = hom(ℳʹ[n])
    𝐔ₙ = (𝐦 ⊗ 𝐦ʹ)
    ∂ₓ𝐮ₙ = [(𝐞₁ ⊗ 𝐦ʹ) (𝐞₂ ⊗ 𝐦ʹ) (𝐦 ⊗ 𝐞₁) (𝐦 ⊗ 𝐞₂)]
    𝐁ₙ = ∂ₓ𝐮ₙ * 𝚲ₙ * ∂ₓ𝐮ₙ'
    𝚺ₙ = 𝛉' * 𝐁ₙ * 𝛉
    𝚺ₙ⁻¹ = inv(𝚺ₙ)
    𝐓₁ = @SMatrix zeros(Float64, l, l)
    for k in 1:l
      𝐞ₖ = 𝐈ₗ[:, k]
      ∂𝐞ₖ𝚺ₙ = (𝐈ₘ ⊗ 𝐞ₖ') * 𝐁ₙ * (𝐈ₘ ⊗ 𝛉) + (𝐈ₘ ⊗ 𝛉') * 𝐁ₙ * (𝐈ₘ ⊗ 𝐞ₖ)
      # Accumulating the result in 𝐓₁ allocates memory,
      # even though the two terms in the
      # summation are both SArrays.
      𝐓₁ = 𝐓₁ + 𝐔ₙ * 𝚺ₙ⁻¹ * (∂𝐞ₖ𝚺ₙ) * 𝚺ₙ⁻¹ * 𝐔ₙ' * 𝛉 * 𝐞ₖ'
    end
    𝐓 = 𝐓 + 𝐓₁
  end
  𝐓
end
