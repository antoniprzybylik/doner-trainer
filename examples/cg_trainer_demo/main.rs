use libdt::layer::Layer;
use libdt::layer::LinLayer;
use libdt::layer::SigmaLayer;
use libdt::network::Network;
use libdt::trainer::CGTrainer;
use libdt::trainer::Trainer;
use libdt::macros::neural_network;

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
                        xd_pair.0.clone().as_slice()));
        d_values.push(DVector::from_column_slice(
                        xd_pair.1.clone().as_slice()));
    }

    let p: Vec<f64> = NiceNetwork::default_initial_params();

    let nn = NiceNetwork::new();
    let mut trainer = CGTrainer::new(
        nn, p, x_values, d_values);

    for i in 1..=1000 {
        trainer.make_step();
        println!("Step {i}; cost: {}", trainer.cost());
    }

    println!("\nFinal parameters:\n{:?}", trainer.params());
}
