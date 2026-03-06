//! MOVT — Multi-Plane Orthogonal Value Transport
//!
//! ## Mechanism
//!
//! Instead of a single Givens rotation angle per (offset, head), MOVT learns
//! r angles on r disjoint dimension pairs.  The transport operator is:
//!
//! ```text
//! T = G_{P_r}(φ_r) · G_{P_{r-1}}(φ_{r-1}) · ··· · G_{P_1}(φ_1)
//! ```
//!
//! where each G_{P_m}(φ) is the identity matrix modified on a single 2×2 block
//! corresponding to the plane (p_m, q_m):
//!
//! ```text
//!   G[p_m, p_m] =  cos(φ)     G[p_m, q_m] = −sin(φ)
//!   G[q_m, p_m] =  sin(φ)     G[q_m, q_m] =  cos(φ)
//! ```
//!
//! Because each G is orthogonal and the product of orthogonal matrices is
//! orthogonal, T is guaranteed orthogonal regardless of the angle values.
//! This ensures exact norm preservation of the transported value vector.
//!
//! ## Relationship to condU-phase (r = 1 case)
//!
//! The current condU-phase scalar mechanism uses a single Givens rotation
//! on one fixed plane per (offset, head).  This is the r = 1 special case
//! of MOVT.  Increasing r allows richer value transformations while
//! maintaining the orthogonality guarantee that prevents norm explosion
//! during multi-layer value transport.
//!
//! ## What this module verifies
//!
//! 1. Orthogonality: T^T · T = I  (within 1e-10 tolerance)
//! 2. Norm preservation: ||T · v|| = ||v||  (within 1e-10)
//! 3. Zero-init identity: all φ = 0 → T = I exactly
//! 4. Causality: T is pointwise on the vector dimension (structural guarantee)
//! 5. Plane independence: distinct angle sets produce distinct transport matrices
//! 6. Composition order: non-commutative for overlapping planes,
//!    commutative for disjoint planes

const DIMENSION: usize = 64;

type Plane = (usize, usize);

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

fn random_angles(seed: u64, count: usize) -> Vec<f64> {
    let mut state = seed;
    (0..count).map(|_| (lcg(&mut state) - 0.5) * 2.0 * std::f64::consts::PI).collect()
}

/// Build the D×D transport matrix T = G_{P_r}(φ_r) ··· G_{P_1}(φ_1).
///
/// Each Givens rotation is left-multiplied onto the accumulating matrix,
/// so the final result applies G_{P_1} first when acting on a vector.
fn build_transport_matrix(planes: &[Plane], angles: &[f64]) -> Vec<f64> {
    assert_eq!(planes.len(), angles.len());
    let mut matrix = vec![0.0; DIMENSION * DIMENSION];
    for index in 0..DIMENSION {
        matrix[index * DIMENSION + index] = 1.0;
    }
    for (plane, &angle) in planes.iter().zip(angles.iter()) {
        let (row_p, row_q) = *plane;
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

/// Apply T = G_{P_r}(φ_r) ··· G_{P_1}(φ_1) directly to a vector
/// without constructing the full matrix.
fn apply_transport(vector: &[f64], planes: &[Plane], angles: &[f64]) -> Vec<f64> {
    assert_eq!(planes.len(), angles.len());
    let mut result = vector.to_vec();
    for (plane, &angle) in planes.iter().zip(angles.iter()) {
        let (p, q) = *plane;
        let cosine = angle.cos();
        let sine = angle.sin();
        let value_p = result[p];
        let value_q = result[q];
        result[p] = cosine * value_p - sine * value_q;
        result[q] = sine * value_p + cosine * value_q;
    }
    result
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

fn matvec(matrix: &[f64], vector: &[f64]) -> Vec<f64> {
    (0..DIMENSION)
        .map(|i| (0..DIMENSION).map(|j| matrix[i * DIMENSION + j] * vector[j]).sum())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLANES: [Plane; 4] = [(0, 1), (2, 3), (4, 5), (6, 7)];

    #[test]
    fn orthogonality_random_angles() {
        let angles = random_angles(42, PLANES.len());
        let transport = build_transport_matrix(&PLANES, &angles);
        let product = matrix_multiply(&transpose(&transport), &transport);

        println!("\n[movt_orthogonal_transport] Orthogonality: T^T · T = I");
        let mut max_off_diagonal = 0.0_f64;
        let mut max_diagonal_error = 0.0_f64;
        for i in 0..DIMENSION {
            for j in 0..DIMENSION {
                let expected = if i == j { 1.0 } else { 0.0 };
                let error = (product[i * DIMENSION + j] - expected).abs();
                if i == j {
                    max_diagonal_error = max_diagonal_error.max(error);
                } else {
                    max_off_diagonal = max_off_diagonal.max(error);
                }
            }
        }
        println!("  Max diagonal error:     {max_diagonal_error:.2e}");
        println!("  Max off-diagonal error: {max_off_diagonal:.2e}");
        assert!(
            max_diagonal_error < 1e-10,
            "Diagonal of T^T·T deviates from 1 by {max_diagonal_error:.2e}"
        );
        assert!(
            max_off_diagonal < 1e-10,
            "Off-diagonal of T^T·T deviates from 0 by {max_off_diagonal:.2e}"
        );
        println!("  ✓ T^T · T = I within 1e-10");
    }

    #[test]
    fn norm_preservation_random_vectors() {
        let angles = random_angles(42, PLANES.len());
        println!("\n[movt_orthogonal_transport] Norm preservation: ||T·v|| = ||v||");
        for seed in 100..110 {
            let vector = random_vector(seed);
            let original_norm = norm(&vector);
            let transported = apply_transport(&vector, &PLANES, &angles);
            let transported_norm = norm(&transported);
            let error = (transported_norm - original_norm).abs();
            println!(
                "  seed={seed}: ||v||={original_norm:.6}, ||T·v||={transported_norm:.6}, Δ={error:.2e}"
            );
            assert!(error < 1e-10, "Norm changed by {error:.2e} for seed {seed}");
        }
        println!("  ✓ Norm preserved within 1e-10 across 10 random vectors");
    }

    #[test]
    fn zero_init_yields_identity() {
        let angles = vec![0.0; PLANES.len()];
        let transport = build_transport_matrix(&PLANES, &angles);

        println!("\n[movt_orthogonal_transport] Zero-init identity: φ=0 → T=I");
        for i in 0..DIMENSION {
            for j in 0..DIMENSION {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(
                    transport[i * DIMENSION + j], expected,
                    "T[{i},{j}] should be {expected} when all angles are zero"
                );
            }
        }
        println!("  ✓ All φ=0 produces exact identity matrix");
    }

    /// T operates pointwise on the value vector's dimensions: it reads v[p] and
    /// v[q] for each plane (p, q) and writes back rotated values to those same
    /// indices.  There is no coupling to any positional index n.  The transport
    /// at position n depends only on the angles (which are per-offset, not
    /// per-position in the MOVT formulation) and v[n−δ].  No information from
    /// positions > n can leak through T.
    #[test]
    fn causality_structural_guarantee() {
        let angles = random_angles(77, PLANES.len());
        let vector = random_vector(200);
        let transported = apply_transport(&vector, &PLANES, &angles);

        assert_eq!(transported.len(), DIMENSION);

        let transport_matrix = build_transport_matrix(&PLANES, &angles);
        let matrix_result = matvec(&transport_matrix, &vector);
        let agreement_error: f64 = transported
            .iter()
            .zip(matrix_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            agreement_error < 1e-10,
            "Direct and matrix-based transport must agree; max error = {agreement_error:.2e}"
        );

        println!("\n[movt_orthogonal_transport] Causality: structural guarantee");
        println!("  T is a D×D matrix acting pointwise on the value vector.");
        println!("  No positional indexing beyond the input vector v[n−δ].");
        println!("  Direct vs matrix application agreement: {agreement_error:.2e}");
        println!("  ✓ Causality holds by construction (pointwise on dimension)");
    }

    #[test]
    fn plane_independence_distinct_angles() {
        let angles_a = random_angles(42, PLANES.len());
        let angles_b = random_angles(99, PLANES.len());
        let transport_a = build_transport_matrix(&PLANES, &angles_a);
        let transport_b = build_transport_matrix(&PLANES, &angles_b);

        let max_difference: f64 = transport_a
            .iter()
            .zip(transport_b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        println!("\n[movt_orthogonal_transport] Plane independence");
        println!("  Angles A: {:?}", &angles_a);
        println!("  Angles B: {:?}", &angles_b);
        println!("  Max matrix difference: {max_difference:.6}");
        assert!(
            max_difference > 1e-6,
            "Different angles must produce different transport matrices; max_diff = {max_difference:.2e}"
        );
        println!("  ✓ Distinct angle assignments produce distinct transport matrices");
    }

    #[test]
    fn composition_order_overlapping_vs_disjoint() {
        let angle_1 = 0.7;
        let angle_2 = 1.3;
        let vector = random_vector(55);

        println!("\n[movt_orthogonal_transport] Composition order");

        // Overlapping planes: (0,1) and (1,2) share index 1
        let overlapping_plane_a: Plane = (0, 1);
        let overlapping_plane_b: Plane = (1, 2);

        let forward = apply_transport(
            &apply_transport(&vector, &[overlapping_plane_a], &[angle_1]),
            &[overlapping_plane_b],
            &[angle_2],
        );
        let reversed = apply_transport(
            &apply_transport(&vector, &[overlapping_plane_b], &[angle_2]),
            &[overlapping_plane_a],
            &[angle_1],
        );
        let overlapping_difference: f64 = forward
            .iter()
            .zip(reversed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        println!(
            "  Overlapping planes (0,1)↔(1,2): max |forward−reversed| = {overlapping_difference:.6}"
        );
        assert!(
            overlapping_difference > 1e-6,
            "Overlapping planes should be non-commutative; diff = {overlapping_difference:.2e}"
        );
        println!("  ✓ Non-commutative for overlapping planes");

        // Disjoint planes: (0,1) and (2,3) share no indices
        let disjoint_plane_a: Plane = (0, 1);
        let disjoint_plane_b: Plane = (2, 3);

        let forward_disjoint = apply_transport(
            &apply_transport(&vector, &[disjoint_plane_a], &[angle_1]),
            &[disjoint_plane_b],
            &[angle_2],
        );
        let reversed_disjoint = apply_transport(
            &apply_transport(&vector, &[disjoint_plane_b], &[angle_2]),
            &[disjoint_plane_a],
            &[angle_1],
        );
        let disjoint_difference: f64 = forward_disjoint
            .iter()
            .zip(reversed_disjoint.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        println!(
            "  Disjoint planes (0,1)↔(2,3): max |forward−reversed| = {disjoint_difference:.2e}"
        );
        assert!(
            disjoint_difference < 1e-14,
            "Disjoint planes should commute exactly; diff = {disjoint_difference:.2e}"
        );
        println!("  ✓ Commutative for disjoint planes");
    }
}
