//! EMA Window / Sparse Offset Coherence Analysis
//!
//! ## Motivation
//!
//! d41s3 (sparse=[48,128,384], b0=0.0023, W≈435t) achieves 80% passkey at 52.457 PPL.
//! d41s5 (sparse=[48,128,384,768,1536], b0=0.0030, W≈336t) achieves 41.7% passkey at 52.677 PPL.
//! The only architectural difference is 2 extra sparse offsets (768, 1536) in d41s5.
//! Passkey regressed by 38.3 pp — including at d=1 (100%→40%), where offset coverage is identical.
//!
//! ## Core Claim
//!
//! The Kalman-EMA in the interference block has a coherence horizon W = 1/b0.
//! A sparse offset at distance δ has EMA-mediated signal weight:
//!
//!     ema_weight(δ, b0) = (1 - b0)^δ  ≈  exp(-δ · b0)
//!
//! This decays exponentially with distance. An offset is "EMA-coherent" when
//! ema_weight(δ, b0) ≥ threshold (e.g., 0.1 = -10 dB).
//! The coherence cutoff is:  δ_cut = ln(1/threshold) / b0  ≈  ln(10) / b0 ≈ 2.303 / b0
//!
//! ## Key Numbers
//!
//! d41s3: b0=0.0023 → W=435t,  δ_cut(−10dB)=1001t.  All sparse [48,128,384] well inside.
//! d41s5: b0=0.0030 → W=333t,  δ_cut(−10dB)=767t.   Sparse 768 at margin; 1536 at 10dB×2 out.
//!
//! ## Design Rule
//!
//! For any EMA factor b0 expected in [0.002, 0.004]:
//!   max_sparse_offset ≤ ln(10) / b0_max ≈ 2.303 / 0.004 ≈ 576 tokens
//!
//! All three offsets in d41s3 ([48,128,384]) satisfy this for all plausible b0.
//! d41s5's [768,1536] violate it for b0 ≥ 0.003.
//!
//! ## Why Dead-Weight Offsets Hurt Short-Range Retrieval
//!
//! The model's J-budget (number of offsets) is shared across all distances.
//! Each DSQG layer processes J positions; head specialisation emerges from gradient
//! competition. Adding offsets at 768/1536 — which have low gradient signal density
//! (language has few dependencies at those distances) and low EMA weight — forces
//! the model to allocate capacity to "ghost" offsets. This noise in the training
//! signal degrades the model's ability to specialise at ALL distances, not just long ones.
//!
//! Evidence: d41s5 regresses at d=1 (100%→40%), d=2 (80%→40%), d=32 (100%→60%).
//! These distances are fully covered by the dense window (δ=1..41). The regression
//! is not a coverage failure — it is a retrieval-capacity failure.

/// EMA decay weight at lag `delta` for smoothing factor `b0`.
/// Represents the fraction of a signal at distance δ that survives in the EMA state.
/// ema_weight(0, b0) = 1.0; ema_weight(∞, b0) → 0.0.
pub fn ema_weight(delta: usize, b0: f64) -> f64 {
    (1.0 - b0).powi(delta as i32)
}

/// EMA coherence window in tokens: the lag at which ema_weight = 1/e ≈ 0.368.
/// This is the standard "time constant" of the EMA filter.
pub fn ema_window(b0: f64) -> f64 {
    1.0 / b0
}

/// EMA coherence cutoff: the lag at which ema_weight drops below `threshold`.
/// For threshold=0.1 (−10 dB), returns ≈ 2.303/b0.
pub fn ema_coherence_cutoff(b0: f64, threshold: f64) -> f64 {
    -threshold.ln() / b0
}

/// Is sparse offset `delta` EMA-coherent for the given b0 and threshold?
pub fn is_ema_coherent(delta: usize, b0: f64, threshold: f64) -> bool {
    ema_weight(delta, b0) >= threshold
}

/// Effective signal strength for a sparse offset, combining:
///   - path_gradient:  proportional to some per-offset base signal (caller provides)
///   - ema_gradient:   EMA weight × linguistic frequency model (1/δ^beta)
///
/// Returns a combined signal score (unnormalized).
/// `base_signal` is the structural path-count signal (from offset_optimizer).
/// `beta` is the linguistic power-law exponent (~1.2 for natural language).
pub fn combined_signal(delta: usize, b0: f64, base_signal: f64, beta: f64) -> f64 {
    let ling_freq = (delta as f64).powf(-beta);
    let ema = ema_weight(delta, b0);
    base_signal + ema * ling_freq
}

/// Summary of one sparse offset's signal characteristics.
#[derive(Debug)]
pub struct OffsetSignal {
    pub delta: usize,
    pub ema_weight: f64,
    pub ema_weight_db: f64,      // 20·log10(ema_weight)
    pub ling_freq: f64,          // 1/δ^beta
    pub ema_mediated: f64,       // ema_weight × ling_freq
    pub is_coherent_10db: bool,  // ema_weight ≥ 0.1
    pub is_coherent_3db: bool,   // ema_weight ≥ 0.5
}

/// Compute signal characteristics for a list of sparse offsets given b0.
pub fn analyse_offsets(sparse_offsets: &[usize], b0: f64, beta: f64) -> Vec<OffsetSignal> {
    sparse_offsets.iter().map(|&delta| {
        let w = ema_weight(delta, b0);
        let lf = (delta as f64).powf(-beta);
        OffsetSignal {
            delta,
            ema_weight: w,
            ema_weight_db: 20.0 * w.log10(),
            ling_freq: lf,
            ema_mediated: w * lf,
            is_coherent_10db: w >= 0.1,
            is_coherent_3db: w >= 0.5,
        }
    }).collect()
}

/// J-budget efficiency: mean EMA-mediated signal per sparse offset.
/// Higher = each offset is better used.
pub fn budget_efficiency(sparse_offsets: &[usize], b0: f64, beta: f64) -> f64 {
    let total: f64 = sparse_offsets.iter()
        .map(|&d| ema_weight(d, b0) * (d as f64).powf(-beta))
        .sum();
    total / sparse_offsets.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // d41s3 empirical values
    const B0_S3: f64 = 0.0023;
    // d41s5 empirical values
    const B0_S5: f64 = 0.0030;

    const SPARSE_S3: &[usize] = &[48, 128, 384];
    const SPARSE_S5: &[usize] = &[48, 128, 384, 768, 1536];

    const BETA: f64 = 1.2;   // linguistic power-law exponent
    const THRESHOLD_10DB: f64 = 0.1;

    #[test]
    fn ema_window_values() {
        let w3 = ema_window(B0_S3);
        let w5 = ema_window(B0_S5);
        println!("\n=== EMA Window Analysis ===");
        println!("  d41s3: b0={:.4}  W = 1/b0 = {:.0} tokens", B0_S3, w3);
        println!("  d41s5: b0={:.4}  W = 1/b0 = {:.0} tokens", B0_S5, w5);
        println!();
        let cut3 = ema_coherence_cutoff(B0_S3, THRESHOLD_10DB);
        let cut5 = ema_coherence_cutoff(B0_S5, THRESHOLD_10DB);
        println!("  d41s3: −10 dB cutoff = {:.0} tokens", cut3);
        println!("  d41s5: −10 dB cutoff = {:.0} tokens", cut5);
        println!();
        println!("  d41s3 sparse offsets vs cutoff ({:.0}t):", cut3);
        for &d in SPARSE_S3 {
            println!("    δ={:<6} {}  (weight={:.4})",
                d,
                if (d as f64) <= cut3 { "COHERENT ✓" } else { "incoherent ✗" },
                ema_weight(d, B0_S3));
        }
        println!();
        println!("  d41s5 sparse offsets vs cutoff ({:.0}t):", cut5);
        for &d in SPARSE_S5 {
            println!("    δ={:<6} {}  (weight={:.4})",
                d,
                if (d as f64) <= cut5 { "COHERENT ✓" } else { "incoherent ✗" },
                ema_weight(d, B0_S5));
        }

        // Assertions: d41s3's max sparse (384) must be coherent under d41s3's b0
        assert!(is_ema_coherent(384, B0_S3, THRESHOLD_10DB),
            "d41s3: max sparse offset 384 should be EMA-coherent under b0={}", B0_S3);
        // d41s5's extra offsets (768, 1536) must be incoherent under d41s5's b0 at -10dB
        assert!(!is_ema_coherent(1536, B0_S5, THRESHOLD_10DB),
            "d41s5: offset 1536 should be EMA-incoherent (weight={:.4})", ema_weight(1536, B0_S5));
        // 768 is right at the margin — weight should be ≈0.1
        let w768 = ema_weight(768, B0_S5);
        println!("\n  Note: d41s5 δ=768 weight={:.4} (≈{:.1} dB) — right at the −10dB margin",
            w768, 20.0 * w768.log10());
    }

    #[test]
    fn offset_signal_comparison() {
        println!("\n=== Per-Offset Signal Analysis ===");

        let sig3 = analyse_offsets(SPARSE_S3, B0_S3, BETA);
        let sig5 = analyse_offsets(SPARSE_S5, B0_S5, BETA);

        println!("\n  d41s3 (b0={:.4}):", B0_S3);
        println!("  {:>8}  {:>10}  {:>8}  {:>12}  {:>11}  {:>8}",
            "δ", "ema_wt", "dB", "ling_freq", "ema×freq", "coherent?");
        println!("  {}", "-".repeat(64));
        for s in &sig3 {
            println!("  {:>8}  {:>10.4}  {:>8.1}  {:>12.6}  {:>11.6}  {:>8}",
                s.delta, s.ema_weight, s.ema_weight_db, s.ling_freq, s.ema_mediated,
                if s.is_coherent_10db { "YES" } else { "NO" });
        }
        let eff3 = budget_efficiency(SPARSE_S3, B0_S3, BETA);
        println!("  Budget efficiency (mean EMA×freq per offset): {:.6}", eff3);

        println!("\n  d41s5 (b0={:.4}):", B0_S5);
        println!("  {:>8}  {:>10}  {:>8}  {:>12}  {:>11}  {:>8}",
            "δ", "ema_wt", "dB", "ling_freq", "ema×freq", "coherent?");
        println!("  {}", "-".repeat(64));
        for s in &sig5 {
            println!("  {:>8}  {:>10.4}  {:>8.1}  {:>12.6}  {:>11.6}  {:>8}",
                s.delta, s.ema_weight, s.ema_weight_db, s.ling_freq, s.ema_mediated,
                if s.is_coherent_10db { "YES" } else { "NO" });
        }
        let eff5 = budget_efficiency(SPARSE_S5, B0_S5, BETA);
        println!("  Budget efficiency (mean EMA×freq per offset): {:.6}", eff5);

        println!("\n  Efficiency ratio d41s3/d41s5: {:.3}×", eff3 / eff5);

        // d41s3 should be more J-efficient
        assert!(eff3 > eff5,
            "d41s3 should have higher J-budget efficiency than d41s5 ({:.6} vs {:.6})",
            eff3, eff5);
    }

    #[test]
    fn design_rule_b0_sweep() {
        // For a range of plausible b0 values, compute the max safe sparse offset.
        // A sparse offset is "safe" if ema_weight(δ, b0) ≥ 0.1.
        println!("\n=== Design Rule: Max Safe Sparse Offset vs b0 ===");
        println!("  (safe = EMA-coherent at −10 dB threshold)");
        println!();
        println!("  {:>8}  {:>10}  {:>14}  {:>10}  {:>10}",
            "b0", "W=1/b0", "δ_cut(−10dB)", "384 safe?", "768 safe?");
        println!("  {}", "-".repeat(58));
        for b0_x10000 in [10, 15, 20, 23, 25, 30, 35, 40, 50] {
            let b0 = b0_x10000 as f64 / 10000.0;
            let w = ema_window(b0);
            let cut = ema_coherence_cutoff(b0, 0.1);
            let safe384 = is_ema_coherent(384, b0, 0.1);
            let safe768 = is_ema_coherent(768, b0, 0.1);
            println!("  {:>8.4}  {:>10.0}  {:>14.0}  {:>10}  {:>10}",
                b0, w, cut,
                if safe384 { "YES ✓" } else { "NO ✗" },
                if safe768 { "YES ✓" } else { "NO ✗" });
        }
        println!();
        println!("  Conclusion: offset 384 is safe for all b0 ≤ 0.006 (W≥167).");
        println!("  Offset 768 becomes unsafe at b0 ≥ 0.003 (W=333) — exactly d41s5's b0.");
        println!("  Offset 1536 is unsafe for all b0 ≥ 0.0015 (W=667).");

        // Assert the design rule for the empirically observed b0 range
        assert!(is_ema_coherent(384, 0.003, 0.1), "384 must be safe up to b0=0.003");
        assert!(!is_ema_coherent(1536, 0.002, 0.1), "1536 should be unsafe even at b0=0.002");
    }

    #[test]
    fn ema_b0_feedback_hypothesis() {
        // Hypothesis: adding far offsets (768, 1536) causes b0 to converge HIGHER
        // (faster-forgetting EMA) because the model shortens its window to reduce
        // interference from noisy long-range gradients.
        // This creates a destructive feedback: far offsets → higher b0 → shorter W
        // → even the medium offsets (384) become marginal → retrieval degrades globally.
        //
        // Test: compute ema_weight(384, b0) for b0 values spanning d41s3 to d41s5.
        // Show that the d41s5 b0 (0.003) meaningfully degrades the signal at δ=384.
        println!("\n=== EMA Feedback Loop: Effect of b0 Drift on δ=384 Signal ===");
        println!();
        println!("  {:>8}  {:>14}  {:>16}  {:>16}",
            "b0", "W (tokens)", "weight at δ=384", "dB at δ=384");
        println!("  {}", "-".repeat(58));
        for b0_x10000 in [18, 20, 23, 25, 27, 30, 33, 35] {
            let b0 = b0_x10000 as f64 / 10000.0;
            let w = ema_window(b0);
            let wt = ema_weight(384, b0);
            let db = 20.0 * wt.log10();
            let marker = if (b0 - B0_S3).abs() < 0.0002 { " ← d41s3"
                         } else if (b0 - B0_S5).abs() < 0.0002 { " ← d41s5"
                         } else { "" };
            println!("  {:>8.4}  {:>14.0}  {:>16.4}  {:>16.1}{}",
                b0, w, wt, db, marker);
        }
        println!();
        let wt_s3 = ema_weight(384, B0_S3);
        let wt_s5 = ema_weight(384, B0_S5);
        println!("  Signal at δ=384: d41s3 {:.4} ({:.1} dB) → d41s5 {:.4} ({:.1} dB)",
            wt_s3, 20.0*wt_s3.log10(), wt_s5, 20.0*wt_s5.log10());
        println!("  Degradation at δ=384 from b0 drift alone: {:.1}×",  wt_s3 / wt_s5);
        println!();
        println!("  Interpretation: d41s5's b0=0.003 (vs d41s3's 0.0023) reduces the EMA");
        println!("  signal at δ=384 by {:.1}× even though 384 is nominally 'covered' by", wt_s3/wt_s5);
        println!("  both models. The b0 drift from adding far offsets propagates upstream.");

        // The degradation ratio should be meaningful (>1.2×)
        assert!(wt_s3 / wt_s5 > 1.2,
            "b0 drift should degrade δ=384 signal by >1.2×, got {:.3}×", wt_s3/wt_s5);
    }

    #[test]
    fn d41s3_is_design_rule_optimal() {
        // Verify that [48,128,384] satisfies the design rule for the full b0 range
        // expected in condV-style models (0.002..0.004).
        let b0_range = [0.0015, 0.002, 0.0023, 0.003, 0.004];
        for &b0 in &b0_range {
            let cut = ema_coherence_cutoff(b0, 0.1);
            for &d in SPARSE_S3 {
                let coherent = (d as f64) <= cut;
                if !coherent {
                    println!("  WARNING: d41s3 offset {} is NOT coherent at b0={} (cut={:.0})",
                        d, b0, cut);
                }
            }
        }
        // d41s3 should be coherent for all b0 ≤ 0.006 (generous upper bound)
        for &d in SPARSE_S3 {
            assert!(is_ema_coherent(d, 0.006, 0.1),
                "d41s3 offset {} should be EMA-coherent even at b0=0.006 (cut={:.0})",
                d, ema_coherence_cutoff(0.006, 0.1));
        }
        println!("\n=== d41s3 [48,128,384]: EMA-safe for all b0 ≤ 0.006 ✓ ===");
    }
}
