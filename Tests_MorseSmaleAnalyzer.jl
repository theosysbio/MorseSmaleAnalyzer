using DynamicalSystemsAnalyzer
using DifferentialEquations
using LinearAlgebra
using Plots

"""
Chemical reaction network examples to test for Morse-Smale properties
"""

# Example 1: Simple reversible reaction A ⟷ B
function simple_reversible_crn(x; k1=1.0, k2=0.5)
    # x[1] = concentration of A
    # x[2] = concentration of B
    # Conservation law: x[1] + x[2] = constant
    dxdt = zeros(2)
    dxdt[1] = -k1*x[1] + k2*x[2]  # Rate of change of A
    dxdt[2] = k1*x[1] - k2*x[2]   # Rate of change of B
    return dxdt
end

# Example 2: Bistable switch with mutual inhibition
function bistable_switch(x; n=2.0, k=1.0, decay=0.1)
    # x[1], x[2] = concentrations of two proteins that inhibit each other
    # Hill function model of inhibition
    dxdt = zeros(2)
    dxdt[1] = k/(1 + (x[2]^n)) - decay*x[1]
    dxdt[2] = k/(1 + (x[1]^n)) - decay*x[2]
    return dxdt
end

# Example 3: Oscillating chemical reaction (Brusselator)
function brusselator(x; a=1.0, b=3.0)
    # x[1], x[2] = concentrations
    # a, b are parameters
    dxdt = zeros(2)
    dxdt[1] = a + x[1]^2*x[2] - (b+1)*x[1]
    dxdt[2] = b*x[1] - x[1]^2*x[2]
    return dxdt
end

# Example 4: Repressilator (genetic oscillator with three components)
function repressilator(x; a=1.0, b=4.0, n=2.0)
    # x[1], x[2], x[3] = concentrations of three proteins
    # Each protein represses the next in a cycle
    dxdt = zeros(3)
    dxdt[1] = a/(1 + (x[3]^n)) - b*x[1]
    dxdt[2] = a/(1 + (x[1]^n)) - b*x[2]
    dxdt[3] = a/(1 + (x[2]^n)) - b*x[3]
    return dxdt
end

# Example 5: Bistable enzyme kinetics (with positive feedback)
function bistable_enzyme(x; k1=1.0, k2=0.1, k3=1.0, K=0.5, n=2.0)
    # x[1] = substrate, x[2] = product
    # Substrate conversion to product with positive feedback from product
    dxdt = zeros(2)
    feedback = (1 + (x[2]/K)^n)/(1 + ((x[2]/K)^n)/k3)
    dxdt[1] = -k1 * x[1] * feedback
    dxdt[2] = k1 * x[1] * feedback - k2 * x[2]
    return dxdt
end

# Test functions for each CRN
function test_crn_dynamics()
    # Define testing domains for each CRN
    domain1 = [0.1 5.0; 0.1 5.0]  # Simple reversible
    domain2 = [0.1 5.0; 0.1 5.0]  # Bistable switch
    domain3 = [0.1 5.0; 0.1 5.0]  # Brusselator
    domain4 = [0.1 5.0; 0.1 5.0; 0.1 5.0]  # Repressilator
    domain5 = [0.1 5.0; 0.1 5.0]  # Bistable enzyme
    
    # Define function wrappers
    f1(x) = simple_reversible_crn(x)
    f2(x) = bistable_switch(x, n=4.0)  # Increase Hill coefficient for stronger bistability
    f3(x) = brusselator(x)
    f4(x) = repressilator(x)
    f5(x) = bistable_enzyme(x, n=4.0)  # Increase Hill coefficient for stronger bistability
    
    # Test each CRN
    println("Testing Simple Reversible Reaction:")
    test1 = is_morse_smale(f1, domain1)
    println(" - Is Morse-Smale? ", test1.is_morse_smale)
    println(" - Is Hyperbolic? ", test1.hyperbolic)
    println(" - Number of attractors: ", test1.num_attractors)
    println(" - Critical points: ", test1.critical_points)
    
    println("\nTesting Bistable Switch:")
    test2 = is_morse_smale(f2, domain2)
    println(" - Is Morse-Smale? ", test2.is_morse_smale)
    println(" - Is Hyperbolic? ", test2.hyperbolic)
    println(" - Number of attractors: ", test2.num_attractors)
    println(" - Basins exist? ", test2.basins_exist)
    
    println("\nTesting Brusselator:")
    test3 = is_morse_smale(f3, domain3)
    println(" - Is Morse-Smale? ", test3.is_morse_smale)
    println(" - Is Hyperbolic? ", test3.hyperbolic)
    println(" - Number of periodic orbits: ", length(test3.periodic_orbits))
    
    println("\nTesting Repressilator:")
    test4 = is_morse_smale(f4, domain4)
    println(" - Is Morse-Smale? ", test4.is_morse_smale)
    println(" - Is Hyperbolic? ", test4.hyperbolic)
    
    println("\nTesting Bistable Enzyme:")
    test5 = is_morse_smale(f5, domain5)
    println(" - Is Morse-Smale? ", test5.is_morse_smale)
    println(" - Is Hyperbolic? ", test5.hyperbolic)
    println(" - Number of attractors: ", test5.num_attractors)
    println(" - Basins exist? ", test5.basins_exist)
    
    # Test gradient-like properties
    println("\nTesting if systems are gradient-like:")
    println(" - Simple Reversible: ", is_gradient_like(f1, domain1).is_gradient_like)
    println(" - Bistable Switch: ", is_gradient_like(f2, domain2).is_gradient_like)
    println(" - Brusselator: ", is_gradient_like(f3, domain3).is_gradient_like)
    println(" - Repressilator: ", is_gradient_like(f4, domain4).is_gradient_like)
    println(" - Bistable Enzyme: ", is_gradient_like(f5, domain5).is_gradient_like)
    
    return (test1, test2, test3, test4, test5)
end

# Visualize the bistable switch dynamics with basins of attraction
function visualize_bistable_basins()
    # Create a grid for vector field
    n_grid = 20
    x_range = range(0.1, 5.0, length=n_grid)
    y_range = range(0.1, 5.0, length=n_grid)
    
    # Compute vector field
    u = zeros(n_grid, n_grid)
    v = zeros(n_grid, n_grid)
    
    for i in 1:n_grid, j in 1:n_grid
        result = bistable_switch([x_range[i], y_range[j]], n=4.0)
        u[i,j] = result[1]
        v[i,j] = result[2]
    end
    
    # Normalize for better visualization
    norm_factor = maximum(sqrt.(u.^2 + v.^2))
    u = u ./ norm_factor
    v = v ./ norm_factor
    
    # Find critical points
    f(x) = bistable_switch(x, n=4.0)
    domain = [0.1 5.0; 0.1 5.0]
    critical_points = find_critical_points(f, domain)
    
    # Create plot
    p = quiver(x_range, y_range, quiver=(u, v), title="Bistable Switch: Vector Field and Basins")
    
    # Plot critical points
    if !isempty(critical_points)
        cp_x = [cp[1] for cp in critical_points]
        cp_y = [cp[2] for cp in critical_points]
        scatter!(p, cp_x, cp_y, marker=:circle, color=:red, 
                markersize=6, label="Critical Points")
    end
    
    # Sample trajectories from different initial conditions to show basins
    n_traj = 50
    initial_points = rand(2, n_traj) .* 4.9 .+ 0.1
    
    for i in 1:n_traj
        x0 = initial_points[:,i]
        tspan = (0.0, 20.0)
        prob = ODEProblem((u, p, t) -> bistable_switch(u, n=4.0), x0, tspan)
        sol = solve(prob, Tsit5())
        
        # Determine which basin this belongs to
        final_point = sol.u[end]
        # Simple classification based on which protein dominates at the end
        if final_point[1] > final_point[2]
            plot!(p, [s[1] for s in sol.u], [s[2] for s in sol.u], 
                  linewidth=1.0, alpha=0.5, color=:blue, label=nothing)
        else
            plot!(p, [s[1] for s in sol.u], [s[2] for s in sol.u], 
                  linewidth=1.0, alpha=0.5, color=:green, label=nothing)
        end
    end
    
    return p
end

# Run the tests and visualizations
results = test_crn_dynamics()
basin_plot = visualize_bistable_basins()
display(basin_plot)