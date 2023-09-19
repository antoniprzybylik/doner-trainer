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
struct Test1Network {
    layers: (LinLayer::<1, 2>,
             SigmaLayer::<2>)
}

#[test]
fn test_network_1() {
    let x_values: Vec<DVector<f64>> = vec![
        DVector::from_column_slice(vector![1f64].as_slice()),
        DVector::from_column_slice(vector![3f64].as_slice())];

    let mut p: Vec<f64> = vec![0.5f64, -0.35f64, 2f64, 1f64];
    let mut nn = Test1Network::new();

    nn.forward(&mut p, x_values[0].clone());
    nn.backward(&p);
    let jm = nn.jacobian(&x_values[0]);

    assert_eq!(p, vec![0.5f64, -0.35f64, 2f64, 1f64]);


    let result = matrix![0.07010371f64, 0f64,          0.07010371f64, 0f64;
                           0f64,          0.22534771f64, 0f64,          0.22534771f64];
    assert_eq!(result.ncols(), jm.ncols());
    assert_eq!(result.nrows(), jm.nrows());
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert_float_eq!(result[(i, j)], jm[(i, j)], abs <= 0.000_000_1);
        }
    }
}

#[test]
fn test_network_2() {
    let x_values: Vec<DVector<f64>> = vec![
        DVector::from_column_slice(vector![1f64].as_slice()),
        DVector::from_column_slice(vector![3f64].as_slice())];

    let mut p: Vec<f64> = vec![0.5f64, -0.35f64, 2f64, 1f64];
    let mut nn = Test1Network::new();

    nn.forward(&mut p, x_values[1].clone());
    nn.backward(&p);
    let jm = nn.jacobian(&x_values[1]);

    let result = matrix![0.08535907f64, 0f64,          0.02845302f64, 0f64;
                           0f64,          0.74953144f64, 0f64,          0.2498438f64];
    assert_eq!(result.ncols(), jm.ncols());
    assert_eq!(result.nrows(), jm.nrows());
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert_float_eq!(result[(i, j)], jm[(i, j)], abs <= 0.000_000_1);
        }
    }
}

#[neural_network]
struct Test2Network {
    layers: (LinLayer::<1, 6>,
             SigmaLayer::<6>,
             LinLayer::<6, 1>)
}

#[test]
fn test_network_3() {
    let x_values: Vec<DVector<f64>> = vec![
        DVector::from_column_slice(vector![2f64].as_slice())];

    let mut p: Vec<f64> = vec![1f64, 1f64, 1f64, 1f64, 1f64, 1f64,
                               0f64, 0f64, 0f64, 0f64, 0f64, 0f64,
                               1f64, 1f64, 1f64, 1f64, 1f64, 1f64,
                               7f64];
    let mut nn = Test2Network::new();

    nn.forward(&mut p, x_values[0].clone());
    nn.backward(&p);
    let jm = nn.jacobian(&x_values[0]);

    let result = vector![0.2099871708f64,
                         0.2099871708f64,
                         0.2099871708f64,
                         0.2099871708f64,
                         0.2099871708f64,
                         0.2099871708f64,
                         0.1049935854f64,
                         0.1049935854f64,
                         0.1049935854f64,
                         0.1049935854f64,
                         0.1049935854f64,
                         0.1049935854f64,
                         0.880797078f64,
                         0.880797078f64,
                         0.880797078f64,
                         0.880797078f64,
                         0.880797078f64,
                         0.880797078f64,
                         1f64].transpose();
    assert_eq!(result.ncols(), jm.ncols());
    assert_eq!(result.nrows(), jm.nrows());
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            assert_float_eq!(result[(i, j)], jm[(i, j)], abs <= 0.000_000_1);
        }
    }
}
