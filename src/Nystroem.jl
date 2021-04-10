"""
    Nystroem(x_samples, num_fun, cov_kernel)

Stores the `num_func` eigenvectors and eigenvalues, with which via the
Nyström method the solution of Fredholm integral equations of the
second kind can be approximated regarding the 
the covariance kernel `cov_kernel`. These are constructed from the points
sampled `x_samples`.
The procedure is described in

Shi, Jiaxin and Sun, Shengyang and Zhu, Jun,
*A Spectral Approach to Gradient Estimation for Implicit Distributions*,
Proceedings of the 35th International Conference on Machine Learning,
4651--4660,
2018

See also: [`(nyst::Nystroem)(x)`](@ref)
"""
struct Nystroem{F <: Real, V <: AbstractVector{F}, M <: AbstractMatrix{F}, C <: CovKer}
    num_samples::Integer
    num_func::Integer
    x_samples::M
    eig_vec::M
    eig_val_inv::V
    cov_kernel::C
end

function compute_eig_K(x_samples::M, cov_kernel::C) where {
    F <: Real, M <: AbstractMatrix{F}, C <: CovKer}
    
    K = cov_kernel(x_samples)
    eigen(Symmetric(K))
end

function Nystroem(x_samples::M, num_func::Integer, cov_kernel::C) where {
        F <: Real, M <: AbstractMatrix{F}, C <: CovKer}    
    eig_val, eig_vec = compute_eig_K(x_samples, cov_kernel)
        
    Nystroem{F, typeof(eig_val), M, C}(size(x_samples, 2), num_func, 
        x_samples, eig_vec[:,(end-(num_func-1)):end], eig_val[(end-(num_func-1)):end].^-1, cov_kernel)
end

function Nystroem(x_samples::M, r_bar::F) where {
    F <: Real, M <: AbstractMatrix{F}}    
    
    cov_kernel = SqExp(median(pairwise(Euclidean(), x_samples, dims=2)))
    eig_val, eig_vec = compute_eig_K(x_samples, cov_kernel)

    num_func = findfirst(cumsum(reverse(eig_val)/sum(eig_val)) .> r_bar)
    
    Nystroem{F, typeof(eig_val), M, typeof(cov_kernel)}(size(x_samples, 2), num_func, 
        x_samples, eig_vec[:,(end-(num_func-1)):end], eig_val[(end-(num_func-1)):end].^-1, cov_kernel)
end

"""
    (nyst::Nystroem)(x)

Estimates the corresponding the solution of Fredholm integral equations of the
second kind of `n` at point `x`.
"""
(nyst::Nystroem)(x::W) where {W <: AbstractVector}  = 
    sqrt(nyst.num_samples) .* nyst.eig_val_inv .* 
        ([nyst.cov_kernel(x, nyst.x_samples[:, m]) for m in 1:nyst.num_samples]' * nyst.eig_vec)'
