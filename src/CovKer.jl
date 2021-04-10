abstract type CovKer end

"""
    SqExp(σ)

Stores the bandwidth parameter `σ` of the squared exponential covariance kernel.

See also: [`(s_e::SqExp)(x, y)`](@ref)
"""
struct SqExp{F <: Real} <: CovKer
    neg_inv_double_σ_sq::F
    
    SqExp(σ::F = 1.0) where F <: Real = new{F}(-F(0.5)/(σ^2))
end

"""
    (s_e::SqExp)(x, y)

Evaluates the squared exponential covariance kernel at `(x, y)`.
"""
(s_e::SqExp)(x::V1, y::V2) where {V1 <: AbstractVector, V2 <: AbstractVector} = 
    exp(dot((x - y),(x - y))*s_e.neg_inv_double_σ_sq)

"""
    (ker<:CovKer)(X)

Evaluates the kernel for all `(X_i, X_j)`.
"""

function (ker::C)(x_samples::AbstractMatrix{F}) where {C <: CovKer, F<:Reall} 
    num_samples = size(x_samples, 2)
    K = Array{F, 2}(undef, num_samples, num_samples)
    for i in 1:num_samples
        for j in i:num_samples
            K[i, j] = ker(x_samples[:,i], x_samples[:,j])
        end
    end
    K
end