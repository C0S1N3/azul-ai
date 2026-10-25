//! NNUE smoke tests: feature extraction + forward + serialisation roundtrip.

use crate::features::{extract, FEATURE_DIM};
use crate::net::{Mlp, NnueEvaluator};
use azul_core::fast::FastState;
use azul_search::eval::Evaluator;

#[test]
fn features_extract_len_and_finite() {
    let st = FastState::new(7);
    let f = extract(&st);
    assert_eq!(f.len(), FEATURE_DIM);
    assert!(f.iter().all(|v| v.is_finite()));
    // Scalar context: bias slot 169 must be 1.0.
    assert!(f[169] == 1.0, "bias slot must be 1.0");
}

#[test]
fn mlp_forward_and_serialise_roundtrip() {
    let mut net = Mlp::new(&[FEATURE_DIM, 128, 16, 16, 1]);
    net.init_random(0xC0FFEE);
    let st = FastState::new(7);
    let f = extract(&st);
    let v = net.forward(&f);
    assert!(v.is_finite() && (0.0..=1.0).contains(&v), "sigmoid output in (0,1), got {}", v);

    // Serialisation roundtrip must preserve the exact output.
    let bytes = net.to_bytes();
    let net2 = Mlp::from_bytes(&bytes).expect("deserialise");
    let v2 = net2.forward(&f);
    assert_eq!(v, v2, "roundtrip must preserve forward output");

    // NnueEvaluator adapts to the Evaluator trait.
    let ev = NnueEvaluator::new(net, 100.0);
    let val = ev.eval(&st);
    assert!(val.is_finite() && (-100.0..=100.0).contains(&val));
}

#[test]
fn sigmoid_bounds_output() {
    // Output must stay in (0,1) even for large inputs (not tanh [-1,1]).
    let mut net = Mlp::new(&[FEATURE_DIM, 8, 1]);
    net.init_random(1);
    let st = FastState::new(11);
    let mut f = extract(&st);
    for v in f.iter_mut() {
        *v *= 50.0;
    }
    let v = net.forward(&f);
    assert!((0.0..=1.0).contains(&v), "sigmoid must bound output to (0,1), got {}", v);
}
