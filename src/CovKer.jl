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
(s_e::SqExp{F})(x::V1, y::V2) where {F <: Real, V1 <: Vector, V2 <: Vector} = 
     exp(dot((x - y),(x - y))*s_e.neg_inv_double_σ_sq)