use nalgebra::DVector;
use nalgebra::RowDVector;
use nalgebra::Matrix;
use nalgebra::base::dimension as dim;

use super::super::network::Network;
use super::Trainer;

use super::common::cost;
use super::common::apply_step;
use super::common::choose_step;

/// Simple gradient descent trainer.
pub struct GDTrainer<N: Network>
{
    p: Vec<f64>,
    x_values: Vec<DVector<f64>>,
    d_values: Vec<DVector<f64>>,
    nn: N,
}

fn net_eval<N: Network>(p: &[f64], x_values: &[DVector<f64>])
    -> Vec<DVector<f64>>
{
    let mut y_values: Vec<DVector<f64>> = Vec::new();
    for x in x_values.into_iter() {
        assert_eq!(x.len(), N::NEURONS_IN);
        y_values.push(N::eval(&p, x.clone()));
    }

    y_values
}

impl<N: Network> Trainer<N> for GDTrainer<N> {
    fn new(nn: N, p: Vec<f64>,
           x_values: Vec<DVector<f64>>,
           d_values: Vec<DVector<f64>>) -> Self
    {
        assert_eq!(p.len(), N::PARAMS_CNT);
        assert_eq!(x_values.len(), d_values.len());

        GDTrainer {
            p,
            x_values,
            d_values,
            nn,
        }
    }

    fn make_step(&mut self) {
        let direction = -(self.grad()).clone();

        let step = choose_step::<N>(
            &mut self.p, &self.x_values,
            &self.d_values, direction);
        apply_step(&mut self.p, &step);
    }

    fn cost(&self) -> f64 {
        let y_values =
            net_eval::<N>(self.p.as_slice(),
                          self.x_values.as_slice());
        let y_values = y_values.as_slice();

        cost(y_values, self.d_values.as_slice())
    }

    fn grad(&mut self) -> RowDVector<f64> {
            let mut grad_sum: RowDVector<f64> =
                Matrix::from_element_generic(
                    dim::U1, dim::Dyn(N::NEURONS_OUT), 0f64);
    
            for i in 0..self.x_values.len() {
                let x = &self.x_values[i];
                let d = &self.d_values[i];
    
                let y = self.nn.forward(&self.p, x.clone());
                self.nn.backward(&self.p);
                let jm = self.nn.jacobian(x);
                let g = (y - d).transpose() * jm;
    
                grad_sum += g;
            }
    
            grad_sum
    }

    fn grad_norm(&mut self) -> f64 {
        self.grad().norm()
    }

    fn params(&self) -> &[f64] {
        &self.p
    }
}
