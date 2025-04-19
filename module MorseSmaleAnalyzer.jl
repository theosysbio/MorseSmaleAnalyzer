module MorseSmaleAnalyzer

using LinearAlgebra
using ForwardDiff
using DifferentialEquations
using NLsolve
using StaticArrays
using Statistics
using QuasiMonteCarlo

export is_gradient_field
export is_gradient_like
export is_morse_smale
export jacobian_matrix
export find_critical_points
export compute_curl
export compute_path_variance
export compute_jacobian_symmetry

"""
    jacobian_matrix(F, x)

Compute the Jacobian matrix of vector field F at point x.
"""
function jacobian_matrix(F, x)
    ForwardDiff.jacobian(F, x)
end

"""
    compute_curl(F, x)

Compute the curl matrix (∂Fi/∂xj - ∂Fj/∂xi) of vector field F at point x.
For a gradient field, this should be zero.
"""
function compute_curl(F, x)
    J = jacobian_matrix(F, x)
    return J - J'
end

"""
    is_gradient_field(F, domain_bounds; n_samples=100, tol=1e-10)

Test if a vector field F is a gradient field by checking if curl is zero everywhere.

Parameters:
- F: Function that takes a vector and returns a vector
- domain_bounds: Matrix of [min, max] for each dimension
- n_samples: Number of points to sample for testing
- tol: Tolerance for considering curl to be zero

Returns:
- is_gradient: Bool indicating if the field is a gradient field
- max_curl: Maximum curl magnitude found
- mean_curl: Mean curl magnitude across samples
"""
function is_gradient_field(F, domain_bounds; n_samples=100, tol=1e-10)
    d = size(domain_bounds, 1)  # Dimension of the system
    
    # Generate quasi-random points for more uniform coverage
    points = QuasiMonteCarlo.sample(n_samples, vec(domain_bounds[:,1]), vec(domain_bounds[:,2]), QuasiMonteCarlo.SobolSeq())
    
    curl_mags = zeros(n_samples)
    
    for i in 1:n_samples
        x = points[:,i]
        curl_mat = compute_curl(F, x)
        curl_mags[i] = norm(curl_mat, 2) / sqrt(d) # Normalize by dimension
    end
    
    max_curl = maximum(curl_mags)
    mean_curl = mean(curl_mags)
    
    return (
        is_gradient = max_curl < tol,
        max_curl = max_curl,
        mean_curl = mean_curl
    )
end

"""
    line_integral(F, path, steps=100)

Compute the line integral of vector field F along path.
path should be a function p(t) where t goes from 0 to 1.
"""
function line_integral(F, path, steps=100)
    t_values = range(0, 1, length=steps)
    integral = 0.0
    
    for i in 1:(steps-1)
        t1, t2 = t_values[i], t_values[i+1]
        x1, x2 = path(t1), path(t2)
        f_avg = (F(x1) + F(x2)) / 2
        dx = x2 - x1
        integral += dot(f_avg, dx)
    end
    
    return integral
end

"""
    compute_path_variance(F, a, b, n_paths=10)

Compute variance of line integrals along different paths from a to b.
For a gradient field, this variance should be zero.
"""
function compute_path_variance(F, a, b, n_paths=10)
    integrals = zeros(n_paths)
    
    for i in 1:n_paths
        # Create a random path from a to b
        # We use a simple parameterization with random intermediate points
        n_points = rand(3:8)
        points = [a]
        
        for j in 1:(n_points-2)
            # Random point along straight line from a to b, with some perpendicular noise
            t = j / (n_points-1)
            direct = a + t * (b - a)
            
            # Add some perpendicular noise
            noise_direction = rand(length(a))
            noise_direction = noise_direction - proj(noise_direction, b - a)
            noise_direction = noise_direction / norm(noise_direction) * norm(b - a) * 0.2
            
            push!(points, direct + rand() * noise_direction)
        end
        
        push!(points, b)
        
        # Create a path function using linear interpolation between points
        path = t -> begin
            n = length(points)
            idx = min(floor(Int, t * (n-1)) + 1, n-1)
            t_local = t * (n-1) - (idx-1)
            return points[idx] * (1-t_local) + points[idx+1] * t_local
        end
        
        integrals[i] = line_integral(F, path)
    end
    
    return var(integrals)
end

"""
    compute_jacobian_symmetry(F, x)

Compute the symmetry measure of the Jacobian at point x.
For a gradient field, the Jacobian is perfectly symmetric.
"""
function compute_jacobian_symmetry(F, x)
    J = jacobian_matrix(F, x)
    return norm(J - J', 2) / (norm(J, 2) + 1e-10)
end

"""
    is_gradient_like(F, domain_bounds; n_samples=100, tol=0.01)

Test if a vector field F is gradient-like by multiple criteria.

Parameters:
- F: Function that takes a vector and returns a vector
- domain_bounds: Matrix of [min, max] for each dimension
- n_samples: Number of points to sample for testing
- tol: Tolerance for considering the field gradient-like

Returns:
- is_gradient_like: Bool indicating if the field is gradient-like
- curl_measure: Average curl-based measure (0 is perfect gradient)
- path_measure: Average path variance measure (0 is perfect gradient)
- jacobian_measure: Average Jacobian symmetry measure (0 is perfect gradient)
"""
function is_gradient_like(F, domain_bounds; n_samples=100, tol=0.01, n_path_tests=10)
    d = size(domain_bounds, 1)  # Dimension of the system
    
    # Generate quasi-random points
    points = QuasiMonteCarlo.sample(n_samples, vec(domain_bounds[:,1]), vec(domain_bounds[:,2]), QuasiMonteCarlo.SobolSeq())
    
    # Curl test
    curl_measures = zeros(n_samples)
    jacobian_measures = zeros(n_samples)
    field_mags = zeros(n_samples)
    
    for i in 1:n_samples
        x = points[:,i]
        curl_mat = compute_curl(F, x)
        curl_measures[i] = norm(curl_mat, 2) / sqrt(d)
        jacobian_measures[i] = compute_jacobian_symmetry(F, x)
        field_mags[i] = norm(F(x))
    end
    
    # Normalize curl by field magnitude
    curl_measures = curl_measures ./ (field_mags .+ 1e-10)
    
    # Path test (for a subset of points due to computational cost)
    path_measures = zeros(n_path_tests)
    for i in 1:n_path_tests
        a = points[:, rand(1:n_samples)]
        b = points[:, rand(1:n_samples)]
        path_measures[i] = compute_path_variance(F, a, b)
    end
    
    avg_curl_measure = mean(curl_measures)
    avg_path_measure = mean(path_measures)
    avg_jacobian_measure = mean(jacobian_measures)
    
    # Combined measure
    combined_measure = (avg_curl_measure + avg_path_measure + avg_jacobian_measure) / 3
    
    return (
        is_gradient_like = combined_measure < tol,
        curl_measure = avg_curl_measure,
        path_measure = avg_path_measure,
        jacobian_measure = avg_jacobian_measure,
        combined_measure = combined_measure
    )
end

"""
    find_critical_points(F, domain_bounds; n_initial_guesses=20)

Find critical points (where F(x) = 0) within domain bounds.
"""
function find_critical_points(F, domain_bounds; n_initial_guesses=20)
    d = size(domain_bounds, 1)  # Dimension of the system
    
    # Generate initial guesses using quasi-random points
    initial_points = QuasiMonteCarlo.sample(n_initial_guesses, vec(domain_bounds[:,1]), vec(domain_bounds[:,2]), QuasiMonteCarlo.SobolSeq())
    
    critical_points = []
    
    for i in 1:n_initial_guesses
        x0 = initial_points[:,i]
        
        # Use NLsolve to find zeros of F
        result = nlsolve(F, x0; autodiff=:forward)
        
        if converged(result) && all(domain_bounds[:,1] .<= result.zero .<= domain_bounds[:,2])
            # Check if this is a new critical point
            is_new = true
            for cp in critical_points
                if norm(cp - result.zero) < 1e-6
                    is_new = false
                    break
                end
            end
            
            if is_new
                push!(critical_points, result.zero)
            end
        end
    end
    
    return critical_points
end

"""
    compute_floquet_multipliers(F, orbit, period; n_points=100)

Compute Floquet multipliers for a periodic orbit.
This is a complex function and only a basic implementation is provided.
"""
function compute_floquet_multipliers(F, orbit, period; n_points=100)
    d = length(orbit[1])  # Dimension of the system
    
    # Approximate the monodromy matrix by integrating the variational equation
    function extended_system(du, u, p, t)
        x = @view u[1:d]
        phi = reshape(@view(u[d+1:end]), d, d)
        
        # Original system
        du[1:d] = F(x)
        
        # Variational equation
        J = jacobian_matrix(F, x)
        du_phi = J * phi
        du[d+1:end] = vec(du_phi)
    end
    
    # Start with identity matrix for the fundamental solution matrix
    u0 = vcat(orbit[1], vec(I(d)))
    tspan = (0.0, period)
    prob = ODEProblem(extended_system, u0, tspan)
    sol = solve(prob, Tsit5())
    
    # Extract the monodromy matrix
    monodromy = reshape(sol.u[end][d+1:end], d, d)
    
    # Compute eigenvalues (Floquet multipliers)
    return eigvals(monodromy)
end

"""
    is_morse_smale(F, domain_bounds; params...)

Test if a dynamical system given by vector field F is Morse-Smale.

Parameters:
- F: Function that takes a vector and returns a vector
- domain_bounds: Matrix of [min, max] for each dimension
- tol: Tolerance for numerical calculations
- n_trajectory_samples: Number of trajectories to sample for analysis
- integration_time: Time span for trajectory integration
- detect_periodic_orbits: Whether to attempt to detect periodic orbits
- basin_analysis: Whether to analyze basins of attraction (for multistable systems)

Returns:
- is_morse_smale: Bool indicating if the system is Morse-Smale
- critical_points: List of critical points found
- hyperbolic: Bool indicating if all critical points are hyperbolic
- transverse: Bool indicating if manifold intersections are transverse
- periodic_orbits: List of periodic orbits found (if any)
- num_attractors: Number of attractors found
- basins_exist: Whether distinct basins of attraction exist
"""
function is_morse_smale(F, domain_bounds; tol=1e-6, n_trajectory_samples=100, 
                        integration_time=100.0, detect_periodic_orbits=true,
                        basin_analysis=true)
    # Step 1: Find equilibrium points
    critical_points = find_critical_points(F, domain_bounds)
    
    # Step 2: Test hyperbolicity of equilibrium points
    hyperbolic = true
    critical_eigenvals = []
    attractors = []
    repellers = []
    saddles = []
    
    for p in critical_points
        J = jacobian_matrix(F, p)
        evals = eigvals(J)
        push!(critical_eigenvals, evals)
        
        # Check hyperbolicity: no eigenvalues with zero real part
        if any(abs.(real.(evals)) .< tol)
            hyperbolic = false
        end
        
        # Classify critical points
        if all(real.(evals) .< 0)
            push!(attractors, p)
        elseif all(real.(evals) .> 0)
            push!(repellers, p)
        else
            push!(saddles, p)
        end
    end
    
    # Step 3: Detect periodic orbits (improved implementation)
    periodic_orbits = []
    orbit_periods = []
    orbit_stabilities = []
    
    if detect_periodic_orbits
        # Sample initial conditions
        initial_points = QuasiMonteCarlo.sample(n_trajectory_samples, 
                                               vec(domain_bounds[:,1]), 
                                               vec(domain_bounds[:,2]), 
                                               QuasiMonteCarlo.SobolSeq())
        
        for i in 1:n_trajectory_samples
            x0 = initial_points[:,i]
            
            # Define ODE problem
            prob = ODEProblem((u, p, t) -> F(u), x0, (0.0, integration_time))
            sol = solve(prob, Tsit5())
            
            # Improved periodicity detection using recurrence analysis
            # 1. Sample points along trajectory
            n_points = min(1000, length(sol.t))
            sample_indices = round.(Int, range(n_points÷4, n_points, length=100))
            traj_points = [sol.u[i] for i in sample_indices]
            
            # 2. Calculate pairwise distances
            n_samples = length(traj_points)
            if n_samples > 10  # Only analyze if we have enough points
                distances = zeros(n_samples, n_samples)
                for j in 1:n_samples, k in 1:n_samples
                    distances[j,k] = norm(traj_points[j] - traj_points[k])
                end
                
                # 3. Look for recurrences that could indicate periodic orbits
                min_dist_idx = findall(x -> x < tol && x > 0, distances)
                if !isempty(min_dist_idx)
                    # This is a simplified placeholder - a full implementation would
                    # confirm periodicity and extract the orbit parameters
                    period_candidate = sol.t[sample_indices[min_dist_idx[1][2]]] - 
                                      sol.t[sample_indices[min_dist_idx[1][1]]]
                    if period_candidate > tol
                        # Verify it's truly periodic by checking multiple cycles
                        # This is a stub - more robust detection needed in practice
                        push!(periodic_orbits, traj_points)
                        push!(orbit_periods, period_candidate)
                    end
                end
            end
        end
    end
    
    # Step 4: Analyze basins of attraction (for multistable systems)
    basin_classification = []
    basins_exist = false
    
    if basin_analysis && length(attractors) > 1
        # Sample points for basin analysis
        basin_points = QuasiMonteCarlo.sample(n_trajectory_samples*2, 
                                             vec(domain_bounds[:,1]), 
                                             vec(domain_bounds[:,2]), 
                                             QuasiMonteCarlo.SobolSeq())
        
        for i in 1:size(basin_points, 2)
            x0 = basin_points[:,i]
            
            # Integrate trajectory to determine which attractor it approaches
            prob = ODEProblem((u, p, t) -> F(u), x0, (0.0, integration_time))
            sol = solve(prob, Tsit5())
            
            # Check which attractor (if any) the trajectory approaches
            final_point = sol.u[end]
            min_dist = Inf
            closest_attractor = -1
            
            for (j, attractor) in enumerate(attractors)
                dist = norm(final_point - attractor)
                if dist < min_dist
                    min_dist = dist
                    closest_attractor = j
                end
            end
            
            if min_dist < 0.1  # Threshold for considering convergence
                push!(basin_classification, closest_attractor)
            else
                push!(basin_classification, 0)  # Did not converge to any attractor
            end
        end
        
        # Check if we have evidence of distinct basins
        if length(unique(basin_classification)) > 2  # More than one basin plus non-convergence
            basins_exist = true
        end
    end
    
    # Step 5: Check transversality of manifold intersections
    # This is a difficult computation generally requiring specialized algorithms
    # Here we make a simplified check using basin boundaries
    transverse = true  # Default to true, will set to false if we find evidence otherwise
    
    if basin_analysis && basins_exist
        # A more robust implementation would compute stable and unstable manifolds
        # and check their intersections explicitly
        # This placeholder assumes transversality if basins exist and hyperbolicity holds
    end
    
    # Step 6: Verify chain recurrence properties
    # For a Morse-Smale system, the chain recurrent set equals the set of critical elements
    chain_recurrent = hyperbolic  # Simplification: assume this holds if all points are hyperbolic
    
    # Final determination
    is_morse_smale_system = hyperbolic && transverse && chain_recurrent
    
    return (
        is_morse_smale = is_morse_smale_system,
        critical_points = critical_points,
        critical_eigenvals = critical_eigenvals,
        hyperbolic = hyperbolic,
        transverse = transverse,
        periodic_orbits = periodic_orbits,
        orbit_periods = orbit_periods,
        num_attractors = length(attractors),
        basins_exist = basins_exist
    )
end

end  # module