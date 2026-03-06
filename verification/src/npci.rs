//! NPCI — Norm-Preserving Coupled Injection
//!
//! ## The instability problem with additive K/V injection
//!
//! In the condU family, interference signals are injected into keys and
//! values via additive updates: k' = k + k_delta.  Across many layers,
//! this causes unbounded norm growth: each injection adds energy to the
//! vector, and after L layers the norm can scale as O(√L) or worse.
//! This is the key failure mode that prevents scaling condU beyond 39M.
//!
//! ## Norm-preserving alternative
//!
//! NPCI replaces raw addition with a bounded-angle rotation in the plane
//! spanned by k and the perpendicular component of k_delta:
//!
//! ```text
//! k' = cos(θ) · k + sin(θ) · ||k|| · û_perp
//! ```
//!
//! where û_perp = normalize(k_delta − (k_delta · k̂) · k̂) is the unit
//! vector in the direction of k_delta perpendicular to k.
//!
//! This is a rotation within the (k, û_perp) plane, preserving
//! ||k'|| = ||k|| exactly for any θ:
//!
//! ```text
//! ||k'||² = cos²(θ)||k||² + sin²(θ)||k||²·||û_perp||² + 2cos(θ)sin(θ)||k||(k · û_perp)
//!         = ||k||²(cos²θ + sin²θ) + 0     [since û_perp ⊥ k and ||û_perp|| = 1]
//!         = ||k||²
//! ```
//!
//! ## What this module verifies
//!
//! 1. Exact norm preservation: ||k'|| = ||k|| for any (k, k_delta, θ)
//! 2. Zero-init identity: θ=0 → k'=k exactly
//! 3. Repeated application stability: 100 NPCI updates preserve norm within 1e-6
//! 4. Contrast with additive injection: 100 additive updates → unbounded norm growth
//! 5. Causality: pointwise at position n (structural)
//! 6. Degenerate case: parallel k_delta → graceful skip (returns k unchanged)
//! 7. Quantitative comparison: npci_vs_additive_norm_comparison

const DIMENSION: usize = 64;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(vector: &[f64]) -> f64 {
    dot(vector, vector).sqrt()
}

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / u32::MAX as f64
}

fn random_vector(seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..DIMENSION).map(|_| lcg(&mut state) * 2.0 - 1.0).collect()
}

fn random_unit_vector(seed: u64) -> Vec<f64> {
    let vector = random_vector(seed);
    let length = norm(&vector);
    vector.iter().map(|x| x / length).collect()
}

/// Apply norm-preserving coupled injection.
///
/// Rotates k toward k_delta by angle theta in the plane spanned by k
/// and the perpendicular component of k_delta.  Preserves ||k|| exactly.
///
/// If k_delta is parallel to k (perpendicular component vanishes),
/// the rotation plane is undefined — returns k unchanged.
fn apply_npci(k: &[f64], k_delta: &[f64], theta: f64) -> Vec<f64> {
    let k_norm = norm(k);
    if k_norm < 1e-15 {
        return k.to_vec();
    }

    let k_hat: Vec<f64> = k.iter().map(|x| x / k_norm).collect();
    let parallel_component = dot(k_delta, &k_hat);
    let perpendicular: Vec<f64> = k_delta
        .iter()
        .zip(k_hat.iter())
        .map(|(delta, hat)| delta - parallel_component * hat)
        .collect();
    let perpendicular_norm = norm(&perpendicular);

    // Relative threshold: in D dimensions, catastrophic cancellation in
    // the perpendicular computation produces residuals of order
    // ||k_delta|| · √D · ε_machine ≈ ||k_delta|| · 2e-15.
    // Using k_norm * 1e-10 as threshold provides ample margin.
    if perpendicular_norm < k_norm * 1e-10 {
        return k.to_vec();
    }

    let u_hat_perpendicular: Vec<f64> = perpendicular
        .iter()
        .map(|x| x / perpendicular_norm)
        .collect();

    let cosine = theta.cos();
    let sine = theta.sin();
    k.iter()
        .zip(u_hat_perpendicular.iter())
        .map(|(&ki, &ui)| cosine * ki + sine * k_norm * ui)
        .collect()
}

/// Compare NPCI vs additive injection over multiple steps.
///
/// Starting from a unit vector, applies `number_of_steps` random updates
/// using both methods.  Returns (final_npci_norm, final_additive_norm).
pub fn npci_vs_additive_norm_comparison(number_of_steps: usize) -> (f64, f64) {
    let initial = random_unit_vector(1);
    let mut npci_vector = initial.clone();
    let mut additive_vector = initial;

    let mut state = 777_u64;
    for _ in 0..number_of_steps {
        let delta = random_vector(state);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let theta = (lcg(&mut state) - 0.5) * std::f64::consts::PI;

        npci_vector = apply_npci(&npci_vector, &delta, theta);

        let scaled_delta: Vec<f64> = delta.iter().map(|x| 0.2 * x).collect();
        additive_vector = additive_vector
            .iter()
            .zip(scaled_delta.iter())
            .map(|(k, d)| k + d)
            .collect();
    }

    (norm(&npci_vector), norm(&additive_vector))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_norm_preservation() {
        println!("\n[npci] Norm preservation: ||k'|| = ||k||");
        for seed in 0..20 {
            let k = random_vector(seed * 3);
            let k_delta = random_vector(seed * 3 + 1);
            let mut state = seed as u64 * 7 + 13;
            let theta = (lcg(&mut state) - 0.5) * 2.0 * std::f64::consts::PI;
            let k_prime = apply_npci(&k, &k_delta, theta);
            let original_norm = norm(&k);
            let result_norm = norm(&k_prime);
            let error = (result_norm - original_norm).abs();
            if seed < 5 {
                println!(
                    "  seed={seed}: ||k||={original_norm:.6}, ||k'||={result_norm:.6}, Δ={error:.2e}"
                );
            }
            assert!(
                error < 1e-10,
                "Norm changed by {error:.2e} for seed {seed}"
            );
        }
        println!("  ✓ Norm preserved within 1e-10 across 20 random (k, k_delta, θ) triples");
    }

    #[test]
    fn zero_init_yields_identity() {
        let k = random_vector(42);
        let k_delta = random_vector(43);
        let k_prime = apply_npci(&k, &k_delta, 0.0);

        println!("\n[npci] Zero-init identity: θ=0 → k'=k");
        let max_difference: f64 = k_prime
            .iter()
            .zip(k.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_difference < 1e-15,
            "θ=0 should produce k unchanged; max diff = {max_difference:.2e}"
        );
        println!("  max |k' − k| = {max_difference:.2e}");
        println!("  ✓ θ=0 returns k exactly");
    }

    #[test]
    fn repeated_application_stability() {
        let initial = random_unit_vector(1);
        let initial_norm = norm(&initial);
        let mut current = initial;

        let mut state = 555_u64;
        let number_of_iterations = 100;
        println!("\n[npci] Repeated application: {number_of_iterations} NPCI updates");

        for iteration in 0..number_of_iterations {
            let delta = random_vector(state);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let theta = (lcg(&mut state) - 0.5) * std::f64::consts::PI;
            current = apply_npci(&current, &delta, theta);

            if iteration % 20 == 19 {
                println!(
                    "  step {:>3}: ||k|| = {:.10}",
                    iteration + 1,
                    norm(&current)
                );
            }
        }

        let final_norm = norm(&current);
        let error = (final_norm - initial_norm).abs();
        println!("  Initial norm: {initial_norm:.10}");
        println!("  Final norm:   {final_norm:.10}");
        println!("  Drift:        {error:.2e}");
        assert!(
            error < 1e-6,
            "After {number_of_iterations} NPCI updates, norm drifted by {error:.2e} (limit: 1e-6)"
        );
        println!("  ✓ Norm stable within 1e-6 after {number_of_iterations} sequential NPCI updates");
    }

    #[test]
    fn additive_injection_norm_diverges() {
        let initial = random_unit_vector(1);
        let initial_norm = norm(&initial);
        let mut current = initial;

        let mut state = 555_u64;
        let number_of_iterations = 100;
        println!("\n[npci] Additive injection contrast: {number_of_iterations} additive updates");

        for iteration in 0..number_of_iterations {
            let delta = random_vector(state);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Consume one LCG call to keep state in sync with NPCI test
            let _ = lcg(&mut state);
            let scaled_delta: Vec<f64> = delta.iter().map(|x| 0.2 * x).collect();
            current = current
                .iter()
                .zip(scaled_delta.iter())
                .map(|(k, d)| k + d)
                .collect();

            if iteration % 20 == 19 {
                println!(
                    "  step {:>3}: ||k|| = {:.6}",
                    iteration + 1,
                    norm(&current)
                );
            }
        }

        let final_norm = norm(&current);
        let growth = final_norm / initial_norm;
        println!("  Initial norm: {initial_norm:.6}");
        println!("  Final norm:   {final_norm:.6}");
        println!("  Growth ratio: {growth:.2}x");
        assert!(
            growth > 2.0,
            "Additive injection should cause significant norm growth; ratio = {growth:.2}"
        );
        println!("  ✓ Additive injection causes {growth:.1}x norm growth (unstable)");
    }

    /// NPCI is a pointwise operation: k'[n] depends only on k[n],
    /// k_delta[n], and θ[n].  There is no coupling to any other sequence
    /// position.  The injection at position n cannot access information
    /// from positions > n.
    #[test]
    fn causality_pointwise_operation() {
        let k = random_vector(42);
        let k_delta = random_vector(43);
        let theta = 0.5;
        let k_prime = apply_npci(&k, &k_delta, theta);

        assert_eq!(k_prime.len(), DIMENSION);

        let k_prime_again = apply_npci(&k, &k_delta, theta);
        let agreement: f64 = k_prime
            .iter()
            .zip(k_prime_again.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(agreement < 1e-15, "Deterministic given same inputs");

        println!("\n[npci] Causality");
        println!("  k'[n] = cos(θ)·k[n] + sin(θ)·||k[n]||·û_perp[n]");
        println!("  All inputs (k, k_delta, θ) are local to position n.");
        println!("  ✓ Pointwise operation; no cross-position dependencies");
    }

    #[test]
    fn degenerate_parallel_k_delta() {
        let k = random_vector(42);
        let k_parallel = k.iter().map(|x| 3.7 * x).collect::<Vec<_>>();
        let k_prime = apply_npci(&k, &k_parallel, 1.0);

        let max_difference: f64 = k_prime
            .iter()
            .zip(k.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        println!("\n[npci] Degenerate case: k_delta parallel to k");
        println!("  k_delta = 3.7 · k (exactly parallel)");
        println!("  max |k' − k| = {max_difference:.2e}");
        assert!(
            max_difference < 1e-14,
            "Parallel k_delta should skip rotation; max diff = {max_difference:.2e}"
        );
        println!("  ✓ Parallel k_delta handled gracefully (returns k unchanged)");

        let mut nearly_parallel = k_parallel.clone();
        nearly_parallel[0] += 1e-16;
        let k_prime_near = apply_npci(&k, &nearly_parallel, 1.0);
        let near_norm_error = (norm(&k_prime_near) - norm(&k)).abs();
        println!("  Nearly-parallel: norm error = {near_norm_error:.2e}");
        assert!(
            near_norm_error < 1e-10,
            "Nearly-parallel case must still preserve norm"
        );
        println!("  ✓ Nearly-parallel case preserves norm");
    }

    #[test]
    fn quantitative_npci_vs_additive_comparison() {
        println!("\n[npci] Quantitative NPCI vs additive comparison");
        println!(
            "  {:>8} | {:>14} | {:>14}",
            "steps", "NPCI norm", "additive norm"
        );
        for &steps in &[10, 25, 50, 100] {
            let (npci_norm, additive_norm) = npci_vs_additive_norm_comparison(steps);
            println!(
                "  {:>8} | {:>14.6} | {:>14.6}",
                steps, npci_norm, additive_norm
            );
            assert!(
                (npci_norm - 1.0).abs() < 0.01,
                "NPCI norm should stay near 1.0 after {steps} steps; got {npci_norm}"
            );
        }
        let (final_npci, final_additive) = npci_vs_additive_norm_comparison(100);
        assert!(
            final_additive > 2.0 * final_npci,
            "Additive norm should greatly exceed NPCI norm after 100 steps"
        );
        println!("  ✓ NPCI preserves norm; additive diverges");
    }
}
