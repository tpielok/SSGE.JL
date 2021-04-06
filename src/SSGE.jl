module SSGE

using LinearAlgebra
using ForwardDiff
using Distances
using Statistics

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
struct SSGEstimator{F <: Real, V <: AbstractVector{F}, M <: AbstractMatrix{F}, C <: CovKer}
    ψ::Nystroem{F, V, M, C}
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
    
    SSGEstimator{F, typeof(ψ.eig_val_inv), M, C}(ψ, β)
end

function SSGEstimator(x_samples::M, r_bar::F) where {
    F <: Real, M <: AbstractMatrix{F}} 
    ψ = Nystroem(x_samples, r_bar)

    ∇ψ(x) = ForwardDiff.jacobian(ψ, x)

    β =  ∇ψ(x_samples[:,1])
    for i in 2:ψ.num_samples
    β = β + ∇ψ(x_samples[:,i])
    end
    β = -1/ψ.num_samples .* β 

    SSGEstimator{F, typeof(ψ.eig_val_inv), M, typeof(ψ.cov_kernel)}(ψ, β)
end

"""
    (ssge::SSGEstimator)(x)

Estimates the corresponding Stein gradient of `ssge` at point `x`.
"""
(ssge::SSGEstimator)(x::V) where{V <: AbstractVector} = 
    (ssge.ψ(x)' * ssge.β)'

end # module
