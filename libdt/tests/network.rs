use libdt::layer::Layer;
use libdt::layer::LinLayer;
use libdt::layer::SigmaLayer;
use libdt::network::Network;
use libdt_macros::neural_network;

use nalgebra::DVector;
use nalgebra::DMatrix;
use nalgebra::vector;
use nalgebra::matrix;

use float_eq::assert_float_eq;

#[neural_network]
struct TestNetwork {
    layers: (LinLayer::<1, 2>,
             SigmaLayer::<2>)
}

#[test]
fn test_network() {
    let x_values: Vec<DVector<f64>> = vec![
        DVector::from_column_slice(vector![1f64].as_slice()),
        DVector::from_column_slice(vector![3f64].as_slice())];

    let mut p: Vec<f64> = vec![0.5f64, -0.35f64, 2f64, 1f64];
    let mut nn = TestNetwork::new();

    // Test 1
    println!("Test 1");
    nn.forward(&mut p, x_values[0].clone());
    nn.backward(&p);
    let jm = nn.jacobian(&x_values[0]);

    assert_eq!(p, vec![0.5f64, -0.35f64, 2f64, 1f64]);


    let result_1 = matrix![0.07010371f64, 0f64,          0.07010371f64, 0f64;
                           0f64,          0.22534771f64, 0f64,          0.22534771f64];
    assert_eq!(result_1.ncols(), jm.ncols());
    assert_eq!(result_1.nrows(), jm.nrows());
    for i in 0..result_1.nrows() {
        for j in 0..result_1.ncols() {
            assert_float_eq!(result_1[(i, j)], jm[(i, j)], abs <= 0.000_000_1);
        }
    }

    // Test 2
    println!("Test 2");
    nn.forward(&mut p, x_values[1].clone());
    nn.backward(&p);
    let jm = nn.jacobian(&x_values[1]);

    let result_2 = matrix![0.08535907f64, 0f64,          0.02845302f64, 0f64;
                           0f64,          0.74953144f64, 0f64,          0.2498438f64];
    assert_eq!(result_2.ncols(), jm.ncols());
    assert_eq!(result_2.nrows(), jm.nrows());
    for i in 0..result_2.nrows() {
        for j in 0..result_2.ncols() {
            assert_float_eq!(result_2[(i, j)], jm[(i, j)], abs <= 0.000_000_1);
        }
    }
}
