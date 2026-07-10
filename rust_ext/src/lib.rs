use pyo3::prelude::*;
use rayon::prelude::*;
use sha1::{Digest, Sha1};
use std::collections::{HashMap, HashSet};

/// Trivial health-check: returns "ok".
#[pyfunction]
fn ping() -> &'static str {
    "ok"
}

/// Sum a list of integers, confirming data crosses the FFI boundary.
#[pyfunction]
fn sum_u64(xs: Vec<u64>) -> u64 {
    xs.iter().sum()
}

/// Number of hardware threads available to rayon.
#[pyfunction]
fn rayon_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Parallel MinHash signatures over a corpus.
///
/// For each doc: tokenize by whitespace, build shingle set (contiguous windows
/// of `shingle` tokens, SHA-1 hashed to u32), then compute the MinHash
/// signature using the universal hash family  h_i(x) = (a_i * x + b_i) mod p
/// where p = (1<<61) - 1.
///
/// Returns one Vec<u32> signature per doc (length = num_perm).
#[pyfunction]
fn minhash_signatures(
    docs: Vec<String>,
    coeff_a: Vec<u64>,
    coeff_b: Vec<u64>,
    shingle: usize,
) -> Vec<Vec<u32>> {
    let num_perm = coeff_a.len();
    let p: u128 = (1u128 << 61) - 1; // Mersenne prime

    docs.par_iter()
        .map(|doc| {
            let tokens: Vec<&str> = doc.split_whitespace().collect();

            if tokens.len() < shingle {
                return vec![u32::MAX; num_perm];
            }

            // Build shingle hash set
            let mut shingle_set = HashSet::new();
            for window in tokens.windows(shingle) {
                let joined = window.join(" ");
                let mut hasher = Sha1::new();
                hasher.update(joined.as_bytes());
                let digest = hasher.finalize();
                let h = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]);
                shingle_set.insert(h);
            }

            if shingle_set.is_empty() {
                return vec![u32::MAX; num_perm];
            }

            // Compute MinHash signature
            let mut sig = vec![u32::MAX; num_perm];
            for &h in &shingle_set {
                let h128 = h as u128;
                for i in 0..num_perm {
                    let hv = ((coeff_a[i] as u128 * h128 + coeff_b[i] as u128) % p) as u32;
                    if hv < sig[i] {
                        sig[i] = hv;
                    }
                }
            }
            sig
        })
        .collect()
}

/// Estimate Jaccard similarity from two MinHash signatures.
///
/// Returns the fraction of positions where sig1[i] == sig2[i].
#[pyfunction]
fn estimate_jaccard(sig1: Vec<u32>, sig2: Vec<u32>) -> f64 {
    if sig1.len() != sig2.len() || sig1.is_empty() {
        return 0.0;
    }
    let matches = sig1.iter().zip(sig2.iter()).filter(|(a, b)| a == b).count();
    matches as f64 / sig1.len() as f64
}

/// LSH banding to find candidate near-duplicate pairs between two corpora.
///
/// `bands * rows` must equal the signature length.  For each band, hashes the
/// band slice of each signature into a bucket key, then emits (i, j) pairs
/// where sig_a[i] and sig_b[j] share a bucket in that band.  Parallelizes
/// across bands with rayon.  Returns deduplicated (i, j) pairs.
#[pyfunction]
fn lsh_candidate_pairs(
    sig_a: Vec<Vec<u32>>,
    sig_b: Vec<Vec<u32>>,
    bands: usize,
    rows: usize,
) -> Vec<(u32, u32)> {
    // Collect per-band candidate sets in parallel, then union + dedup.
    let per_band: Vec<Vec<(u32, u32)>> = (0..bands)
        .into_par_iter()
        .map(|band| {
            let start = band * rows;
            let end = start + rows;

            // Build map: band_key -> list of B indices
            let mut b_map: HashMap<Vec<u8>, Vec<u32>> = HashMap::new();
            for (j, sig) in sig_b.iter().enumerate() {
                let key = band_key(&sig[start..end]);
                b_map.entry(key).or_default().push(j as u32);
            }

            // For each A index, look up its band key
            let mut pairs = Vec::new();
            for (i, sig) in sig_a.iter().enumerate() {
                let key = band_key(&sig[start..end]);
                if let Some(js) = b_map.get(&key) {
                    for &j in js {
                        pairs.push((i as u32, j));
                    }
                }
            }
            pairs
        })
        .collect();

    // Dedup across bands
    let mut seen = HashSet::new();
    for pairs in per_band {
        for pair in pairs {
            seen.insert(pair);
        }
    }
    seen.into_iter().collect()
}

/// Hash a band slice (rows u32 values) into a byte key for bucket lookup.
/// Uses SHA-1 truncated to 8 bytes for a compact, collision-resistant key.
fn band_key(slice: &[u32]) -> Vec<u8> {
    let mut hasher = Sha1::new();
    for &v in slice {
        hasher.update(v.to_le_bytes());
    }
    let digest = hasher.finalize();
    digest[..8].to_vec()
}

/// Python module definition.
#[pymodule]
fn aligntune_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(sum_u64, m)?)?;
    m.add_function(wrap_pyfunction!(rayon_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(minhash_signatures, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(lsh_candidate_pairs, m)?)?;
    Ok(())
}
