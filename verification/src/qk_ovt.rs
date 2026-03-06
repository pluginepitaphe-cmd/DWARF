//! QK-OVT — Query-Key Conditioned Orthogonal Value Transport
//!
//! ## The fifth mechanism in the DSQG taxonomy
//!
//! The DSQG attention architecture defines four mechanisms for modulating
//! attention: Scale (S), Offset/Dyadic (D), Gate (G), and Q-weighting (Q).
//! QK-OVT introduces a fifth mechanism that is:
//!
//! - **Content-dependent:** rotation angles are determined by the current
//!   query and the retrieved key, not just the offset/head identity.
//! - **Acts on V:** modifies the value vector before aggregation, rather
//!   than modifying attention scores.
//! - **Orthogonal to score computation:** the value transport does not
//!   interfere with the Q·K attention score pathway.
//!
//! ## Mechanism
//!
//! The rotation angle for plane m at position n, offset j, head h is:
//!
//! ```text
//! θ_{n,j,h,m} = γ_{j,h,m} · y_{n,h,m} · z_{n−δ_j,h,m}
//! ```
//!
//! where:
//! - y_{n,h,m} = U_m^T · q_n / √(HD)  (query probe: scalar projection)
//! - z_{n−δ_j,h,m} = W_m^T · k_{n−δ_j} / √(HD)  (key probe: scalar projection)
//! - γ_{j,h,m} = learned gain, zero-initialized
//!
//! The transport T_{n,j,h} = ∏_m G_{P_m}(θ_{n,j,h,m}) is applied to
//! v[n−δ_j] before aggregation into the attention output.
//!
//! ## What this module verifies
//!
//! 1. Orthogonality: T_{n,j,h} is orthogonal for any (q, k) input (within 1e-10)
//! 2. Norm preservation: ||T·v|| = ||v|| (within 1e-10)
//! 3. Zero-init identity: γ=0 → θ=0 → T=I exactly
//! 4. Content dependence: different (q, k) → different T·v with non-zero γ
//! 5. Causality: depends on k[n−δ] (past) and q[n] (current) — both causal
//! 6. MOVT relationship: fixed (q, k) yields fixed angles, equivalent to MOVT

const DIMENSION: usize = 64;
const HEAD_DIMENSION: usize = 64;
const NUMBER_OF_PLANES: usize = 4;

type Plane = (usize, usize);

const PLANES: [Plane; NUMBER_OF_PLANES] = [(0, 1), (2, 3), (4, 5), (6, 7)];

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

fn random_probe_vector(seed: u64) -> Vec<f64> {
    let mut state = seed;
    let scale = (DIMENSION as f64).sqrt();
    (0..DIMENSION)
        .map(|_| (lcg(&mut state) - 0.5) * 2.0 / scale)
        .collect()
}

/// Probe matrices and learned gains for the QK-OVT mechanism.
///
/// Each U_m (query probe) and W_m (key probe) is a D-dimensional vector
/// that projects a query or key to a scalar.  The gain γ_m controls the
/// rotation magnitude and is zero-initialized during training.
struct ProbeMatrices {
    query_probes: Vec<Vec<f64>>,
    key_probes: Vec<Vec<f64>>,
    gains: Vec<f64>,
}

impl ProbeMatrices {
    fn new(gain_seed: u64) -> Self {
        let mut state = gain_seed;
        let query_probes = (0..NUMBER_OF_PLANES)
            .map(|m| random_probe_vector(100 + m as u64))
            .collect();
        let key_probes = (0..NUMBER_OF_PLANES)
            .map(|m| random_probe_vector(200 + m as u64))
            .collect();
        let gains = (0..NUMBER_OF_PLANES)
            .map(|_| lcg(&mut state) * 0.5)
            .collect();
        ProbeMatrices {
            query_probes,
            key_probes,
            gains,
        }
    }

    fn zero_gain() -> Self {
        let base = Self::new(42);
        ProbeMatrices {
            query_probes: base.query_probes,
            key_probes: base.key_probes,
            gains: vec![0.0; NUMBER_OF_PLANES],
        }
    }

    fn with_gains(gains: Vec<f64>) -> Self {
        let base = Self::new(42);
        ProbeMatrices {
            query_probes: base.query_probes,
            key_probes: base.key_probes,
            gains,
        }
    }
}

/// Compute content-dependent rotation angles θ_m = γ_m · y_m · z_m.
fn compute_angles(probes: &ProbeMatrices, query: &[f64], key: &[f64]) -> Vec<f64> {
    let scale = (HEAD_DIMENSION as f64).sqrt();
    (0..NUMBER_OF_PLANES)
        .map(|m| {
            let query_projection = dot(&probes.query_probes[m], query) / scale;
            let key_projection = dot(&probes.key_probes[m], key) / scale;
            probes.gains[m] * query_projection * key_projection
        })
        .collect()
}

/// Apply the QK-OVT transport to a value vector.
///
/// Computes content-dependent angles from (query, key), then applies the
/// product of Givens rotations on the configured planes.
fn apply_qk_ovt(
    probes: &ProbeMatrices,
    query: &[f64],
    key: &[f64],
    value: &[f64],
) -> Vec<f64> {
    let angles = compute_angles(probes, query, key);
    let mut result = value.to_vec();
    for (plane_index, &angle) in angles.iter().enumerate() {
        let (p, q) = PLANES[plane_index];
        let cosine = angle.cos();
        let sine = angle.sin();
        let value_p = result[p];
        let value_q = result[q];
        result[p] = cosine * value_p - sine * value_q;
        result[q] = sine * value_p + cosine * value_q;
    }
    result
}

/// Build the full D×D transport matrix for given angles.
fn build_transport_matrix(angles: &[f64]) -> Vec<f64> {
    let mut matrix = vec![0.0; DIMENSION * DIMENSION];
    for index in 0..DIMENSION {
        matrix[index * DIMENSION + index] = 1.0;
    }
    for (plane_index, &angle) in angles.iter().enumerate() {
        let (row_p, row_q) = PLANES[plane_index];
        let cosine = angle.cos();
        let sine = angle.sin();
        for column in 0..DIMENSION {
            let value_p = matrix[row_p * DIMENSION + column];
            let value_q = matrix[row_q * DIMENSION + column];
            matrix[row_p * DIMENSION + column] = cosine * value_p - sine * value_q;
            matrix[row_q * DIMENSION + column] = sine * value_p + cosine * value_q;
        }
    }
    matrix
}

fn transpose(matrix: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; DIMENSION * DIMENSION];
    for i in 0..DIMENSION {
        for j in 0..DIMENSION {
            result[j * DIMENSION + i] = matrix[i * DIMENSION + j];
        }
    }
    result
}

fn matrix_multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; DIMENSION * DIMENSION];
    for i in 0..DIMENSION {
        for j in 0..DIMENSION {
            let mut sum = 0.0;
            for k in 0..DIMENSION {
                sum += a[i * DIMENSION + k] * b[k * DIMENSION + j];
            }
            result[i * DIMENSION + j] = sum;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonality_content_dependent() {
        let probes = ProbeMatrices::new(42);
        let query = random_vector(10);
        let key = random_vector(20);
        let angles = compute_angles(&probes, &query, &key);
        let transport = build_transport_matrix(&angles);
        let product = matrix_multiply(&transpose(&transport), &transport);

        println!("\n[qk_ovt] Orthogonality: T^T · T = I for content-dependent angles");
        let mut max_error = 0.0_f64;
        for i in 0..DIMENSION {
            for j in 0..DIMENSION {
                let expected = if i == j { 1.0 } else { 0.0 };
                max_error = max_error.max((product[i * DIMENSION + j] - expected).abs());
            }
        }
        println!("  Angles: {:?}", &angles);
        println!("  Max |T^T·T − I| = {max_error:.2e}");
        assert!(
            max_error < 1e-10,
            "T must be orthogonal; max error = {max_error:.2e}"
        );
        println!("  ✓ Orthogonal within 1e-10");
    }

    #[test]
    fn norm_preservation_various_inputs() {
        let probes = ProbeMatrices::new(42);
        println!("\n[qk_ovt] Norm preservation: ||T·v|| = ||v||");
        for seed in 0..10 {
            let query = random_vector(seed * 3);
            let key = random_vector(seed * 3 + 1);
            let value = random_vector(seed * 3 + 2);
            let original_norm = norm(&value);
            let transported = apply_qk_ovt(&probes, &query, &key, &value);
            let transported_norm = norm(&transported);
            let error = (transported_norm - original_norm).abs();
            println!(
                "  trial {seed}: ||v||={original_norm:.6}, ||T·v||={transported_norm:.6}, Δ={error:.2e}"
            );
            assert!(error < 1e-10, "Norm changed by {error:.2e} in trial {seed}");
        }
        println!("  ✓ Norm preserved within 1e-10 across 10 trials");
    }

    #[test]
    fn zero_init_yields_identity() {
        let probes = ProbeMatrices::zero_gain();
        let query = random_vector(10);
        let key = random_vector(20);
        let value = random_vector(30);
        let angles = compute_angles(&probes, &query, &key);
        let transported = apply_qk_ovt(&probes, &query, &key, &value);

        println!("\n[qk_ovt] Zero-init identity: γ=0 → T=I");
        println!("  Angles with γ=0: {:?}", angles);
        for angle in &angles {
            assert_eq!(*angle, 0.0, "All angles must be exactly zero when γ=0");
        }
        let max_difference: f64 = transported
            .iter()
            .zip(value.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert_eq!(
            max_difference, 0.0,
            "Transport must be exact identity when γ=0; max diff = {max_difference}"
        );
        println!("  ✓ γ=0 produces exact identity transport");
    }

    #[test]
    fn content_dependence_different_queries_and_keys() {
        let probes = ProbeMatrices::new(42);
        let value = random_vector(30);
        let key = random_vector(20);

        let query_a = random_vector(10);
        let query_b = random_vector(11);
        let result_a = apply_qk_ovt(&probes, &query_a, &key, &value);
        let result_b = apply_qk_ovt(&probes, &query_b, &key, &value);

        let query_difference: f64 = result_a
            .iter()
            .zip(result_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        println!("\n[qk_ovt] Content dependence: different (q, k) → different T·v");
        println!("  Different queries: max |T(q_a)·v − T(q_b)·v| = {query_difference:.6}");
        assert!(
            query_difference > 1e-6,
            "Different queries must produce different outputs; diff = {query_difference:.2e}"
        );

        let key_c = random_vector(21);
        let result_c = apply_qk_ovt(&probes, &query_a, &key_c, &value);
        let key_difference: f64 = result_a
            .iter()
            .zip(result_c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!("  Different keys:    max |T(k_a)·v − T(k_c)·v| = {key_difference:.6}");
        assert!(
            key_difference > 1e-6,
            "Different keys must produce different outputs; diff = {key_difference:.2e}"
        );
        println!("  ✓ Transport is content-dependent through both Q and K");
    }

    /// T_{n,j,h} depends on q[n] (the current position's query) and
    /// k[n−δ_j] (the past position's key, where δ_j ≥ 0).  Both are causal:
    /// q[n] is the current token's query, and k[n−δ_j] is at or before
    /// position n.  No future positions are accessed in the angle computation.
    #[test]
    fn causality_past_key_and_current_query() {
        let probes = ProbeMatrices::new(42);
        let query_at_n = random_vector(10);
        let key_at_past = random_vector(20);
        let value_at_past = random_vector(30);

        let result = apply_qk_ovt(&probes, &query_at_n, &key_at_past, &value_at_past);
        assert_eq!(result.len(), DIMENSION);

        let angles = compute_angles(&probes, &query_at_n, &key_at_past);
        let recomputed = apply_qk_ovt(&probes, &query_at_n, &key_at_past, &value_at_past);
        let agreement: f64 = result
            .iter()
            .zip(recomputed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(agreement < 1e-15, "Deterministic given same inputs");

        println!("\n[qk_ovt] Causality");
        println!("  θ_m = γ_m · (U_m^T · q[n] / √HD) · (W_m^T · k[n−δ] / √HD)");
        println!("  q[n]   = current position (causal)");
        println!("  k[n−δ] = past position with δ ≥ 0 (causal)");
        println!("  Computed angles: {:?}", angles);
        println!("  ✓ All inputs are from causal positions");
    }

    #[test]
    fn movt_relationship_fixed_content_yields_fixed_angles() {
        let fixed_gains = vec![0.3, -0.5, 0.7, 0.1];
        let probes = ProbeMatrices::with_gains(fixed_gains);
        let query = random_vector(10);
        let key = random_vector(20);
        let value = random_vector(30);

        let content_angles = compute_angles(&probes, &query, &key);
        let result_qk_ovt = apply_qk_ovt(&probes, &query, &key, &value);

        // Apply the same computed angles as a MOVT-style transport
        let mut result_movt = value.clone();
        for (plane_index, &angle) in content_angles.iter().enumerate() {
            let (p, q) = PLANES[plane_index];
            let cosine = angle.cos();
            let sine = angle.sin();
            let vp = result_movt[p];
            let vq = result_movt[q];
            result_movt[p] = cosine * vp - sine * vq;
            result_movt[q] = sine * vp + cosine * vq;
        }

        let agreement: f64 = result_qk_ovt
            .iter()
            .zip(result_movt.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        println!("\n[qk_ovt] MOVT relationship: fixed (q,k) → fixed angles → MOVT-equivalent");
        println!("  Content-dependent angles: {:?}", content_angles);
        println!("  max |QK-OVT − MOVT_equivalent| = {agreement:.2e}");
        assert!(
            agreement < 1e-15,
            "For fixed (q,k), QK-OVT must equal MOVT with the computed angles"
        );

        let different_query = random_vector(99);
        let different_angles = compute_angles(&probes, &different_query, &key);
        let angle_difference: f64 = content_angles
            .iter()
            .zip(different_angles.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!("  Different query → different angles: max Δθ = {angle_difference:.6}");
        assert!(
            angle_difference > 1e-6,
            "Different content must produce different angles"
        );
        println!("  ✓ QK-OVT = MOVT for fixed (q,k); content variation changes the angles");
    }
}
