"""
    Nystroem(x_samples, num_fun, cov_kernel)

Stores the `num_func` eigenvectors and eigenvalues, with which via the
Nyström method the solution of Fredholm integral equations of the
second kind can be approximated regarding the 
the covariance kernel `cov_kernel. These are constructed from the points
sampled `x_samples`.
The procedure is described in

Shi, Jiaxin and Sun, Shengyang and Zhu, Jun,
*A Spectral Approach to Gradient Estimation for Implicit Distributions*,
Proceedings of the 35th International Conference on Machine Learning,
4651--4660,
2018

See also: [`(nyst::Nystroem)(x)`](@ref)
"""
struct Nystroem{F <: Real, V <: AbstractArray{F, 1}, M <: AbstractArray{F,2}, C <: CovKer}
    num_samples::Integer
    num_func::Integer
    x_samples::M
    eig_vec::M
    eig_val_inv::V
    cov_kernel::C
end

function Nystroem(x_samples::M, num_func::Integer, cov_kernel::C) where {
        F <: Real, M <: AbstractArray{F,2}, C <: CovKer}
    num_samples = size(x_samples, 2)
    
    K = Array{F, 2}(undef, num_samples, num_samples)
    for i in 1:num_samples
        for j in i:num_samples
            K[i, j] = cov_kernel(x_samples[:,i], x_samples[:,j])
        end
    end
    K = Symmetric(K)
    
    eig_vec_l, eig_val, eig_vec_r = tsvd(K, num_func)
    
    Nystroem{F, typeof(eig_val), M, C}(num_samples, num_func, 
        x_samples, eig_vec_r, eig_val.^-1, cov_kernel)
end

"""
    (nyst::Nystroem)(x)

Estimates the corresponding the solution of Fredholm integral equations of the
second kind of `n` at point `x`.
"""
(nyst::Nystroem{F, V, M, C})(x::W) where {F <: Real, V <: AbstractArray{F, 1}, 
    M <: AbstractArray{F,2}, 
    W <: Vector, C <: CovKer}  = 
    sqrt(nyst.num_samples) .* nyst.eig_val_inv .* 
        ([nyst.cov_kernel(x, nyst.x_samples[:, m]) for m in 1:nyst.num_samples]' * nyst.eig_vec)'
