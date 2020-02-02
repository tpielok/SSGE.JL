module SSGE

using LinearAlgebra
using TSVD
using ForwardDiff

include("CovKer.jl")
include("Nystroem.jl")

"""
    SSGEstimator(x_samples, num_eig_func, cov_kernel)

Stores the eigenfunctions `ψ` and the projection matrix `β`, with which
the Stein gradient can be approximated. These are constructed from the points
sampled `x_samples` with `num_eig_func` eigenfunctions regarding the 
the covariance kernel `cov_kernel`.
The procedure is described in

Shi, Jiaxin and Sun, Shengyang and Zhu, Jun,
*A Spectral Approach to Gradient Estimation for Implicit Distributions*,
Proceedings of the 35th International Conference on Machine Learning,
4651--4660,
2018

See also: [`(ssge::SSGEstimator)(x)`](@ref)
"""
struct SSGEstimator{F <: Real, M <: AbstractMatrix{F}}
    ψ::Nystroem
    β::M
end

function SSGEstimator(x_samples::M, num_eig_func::Integer, cov_kernel::C) where {
        F <: Real, M <: AbstractMatrix{F}, C <: CovKer} 
    ψ = Nystroem(x_samples, num_eig_func, cov_kernel)
    
    ∇ψ(x) = ForwardDiff.jacobian(ψ, x)

    β =  ∇ψ(x_samples[:,1])
    for i in 2:ψ.num_samples
       β = β + ∇ψ(x_samples[:,i])
    end
    β = -1/ψ.num_samples .* β 
    
    SSGEstimator{F, M}(ψ, β)
end

"""
    (ssge::SSGEstimator)(x)

Estimates the corresponding Stein gradient of `ssge` at point `x`.
"""
(ssge::SSGEstimator{F, M})(x::V) where{F <: Real, M <: AbstractMatrix{F}, 
    V <: AbstractVector{F}} = 
    (ssge.ψ(x)' * ssge.β)'

end # module
