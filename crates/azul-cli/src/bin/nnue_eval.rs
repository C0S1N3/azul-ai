//! Load a trained NNUE weight file and evaluate a few positions — sanity check
//! that the Rust inference matches the training-side export.

use azul_core::fast::FastState;
use azul_nnue::features::extract;
use azul_nnue::net::Mlp;
use azul_search::eval::Evaluator;
use std::fs;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "out/nnue.bin".to_string());
    let bytes = fs::read(&path).expect("read nnue weights");
    let net = Mlp::from_bytes(&bytes).expect("parse nnue weights");
    println!("loaded {} (dims {:?})", path, net.dims);

    let ev = azul_nnue::NnueEvaluator::new(net, 100.0);
    for seed in [7u64, 42, 99, 123] {
        let st = FastState::new(seed);
        let f = extract(&st);
        let raw = ev.value(&st);
        let scored = ev.eval(&st);
        println!(
            "seed {:>3}: sigmoid {:.4}  eval {:+.2}  (feat[154..159]={:?})",
            seed, raw, scored, &f[154..159]
        );
    }
}
