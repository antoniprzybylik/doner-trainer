use libdt::layer::Layer;
use libdt::layer::LinLayer;
use libdt::layer::SigmaLayer;
use libdt::network::Network;
use libdt::trainer::GDTrainer;
use libdt::trainer::Trainer;
use libdt::macros::neural_network;

use rand::Rng;
use nalgebra::DVector;
use nalgebra::DMatrix;

mod data;

#[neural_network]
struct NiceNetwork {
    layers: (LinLayer::<1, 10>,
             SigmaLayer::<10>,
             LinLayer::<10, 1>)
}

fn main() {
    let mut x_values: Vec<DVector<f64>> =
        Vec::with_capacity(data::XD_PAIRS.len());
    let mut d_values: Vec<DVector<f64>> =
        Vec::with_capacity(data::XD_PAIRS.len());

    for xd_pair in data::XD_PAIRS.iter() {
        x_values.push(DVector::from_column_slice(
                        xd_pair.0.as_slice()));
        d_values.push(DVector::from_column_slice(
                        xd_pair.1.as_slice()));
    }

    let mut rng = rand::thread_rng();
    let mut p: Vec<f64> = Vec::new();
    for _ in 0..NiceNetwork::PARAMS_CNT {
        p.push(rng.gen_range(0.0..1.0));
    }

    let nn = NiceNetwork::new();
    let mut trainer = GDTrainer::new(
        nn, p, x_values, d_values);

    for i in 1..=100 {
        trainer.make_step();
        println!("Step {i}; cost: {}", trainer.cost());
    }

    println!("\nFinal parameters:\n{:?}", trainer.params());
}
