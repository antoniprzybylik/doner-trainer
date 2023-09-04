use nalgebra::SVector;
use nalgebra::RowSVector;

use super::super::network::Network;
use super::Trainer;

use super::common::cost;
use super::common::apply_step;
use super::common::choose_step;

/// Trainer using Conjugate Gradient Method.
pub struct CGTrainer<N,
                     const PARAMS_CNT: usize,
                     const NEURONS_IN: usize,
                     const NEURONS_OUT: usize>
where
    N: Network<PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
{
    p: Vec<f64>,
    x_values: Vec<SVector<f64, NEURONS_IN>>,
    d_values: Vec<SVector<f64, NEURONS_OUT>>,
    nn: N,

    step_num: usize,
    prev_grad: RowSVector<f64, PARAMS_CNT>,
    prev_direction: RowSVector<f64, PARAMS_CNT>,
}

fn net_eval<N,
    const PARAMS_CNT: usize,
    const NEURONS_IN: usize,
    const NEURONS_OUT: usize>(p: &[f64],
    x_values: &[SVector<f64, NEURONS_IN>])
    -> Vec<SVector<f64, NEURONS_OUT>>
where
    N: Network<PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
{
    let mut y_values: Vec<SVector<f64, NEURONS_OUT>> =
        Vec::new();

    for x in x_values.into_iter() {
        y_values.push(N::eval(&p, x.clone()));
    }

    y_values
}

impl<N, const PARAMS_CNT: usize,
     const NEURONS_IN: usize,
     const NEURONS_OUT: usize> 
     Trainer<N, PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
    for CGTrainer<N, PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
where
    N: Network<PARAMS_CNT, NEURONS_IN, NEURONS_OUT>
{
    fn new(nn: N, p: Vec<f64>,
           x_values: Vec<SVector<f64, NEURONS_IN>>,
           d_values: Vec<SVector<f64, NEURONS_OUT>>) -> Self
    {
        assert_eq!(p.len(), PARAMS_CNT);
        assert_eq!(x_values.len(),
                   d_values.len());

        CGTrainer {
            p,
            x_values,
            d_values,
            nn,

            step_num: 0usize,
            prev_grad: nalgebra::zero(),
            prev_direction: nalgebra::zero(),
        }
    }

    fn make_step(&mut self) {
        let grad = self.grad();

        let direction: RowSVector<f64, PARAMS_CNT>;
        if self.step_num % PARAMS_CNT == 0 {
            direction = -grad;
        } else {
            let beta = (grad *
                        (grad - self.prev_grad).transpose())[0] /
                       (self.prev_grad *
                        self.prev_grad.transpose())[0];

            direction = -grad + beta*self.prev_direction;
        }

        let step = choose_step::<N, PARAMS_CNT,
                                 NEURONS_IN, NEURONS_OUT>
            (&mut self.p, &self.x_values, &self.d_values,
             direction.clone());
        apply_step(&mut self.p, &step);

        // Update current state.
        self.step_num += 1;
        self.prev_grad = grad;
        self.prev_direction = direction;
    }

    fn cost(&self) -> f64 {
        let y_values = net_eval::<N, PARAMS_CNT,
                                  NEURONS_IN, NEURONS_OUT>(
                                self.p.as_slice(),
                                self.x_values.as_slice());
        let y_values = y_values.as_slice();

        cost(y_values, self.d_values.as_slice())
    }

    fn grad(&mut self) -> RowSVector<f64, PARAMS_CNT>
    {
            let mut grad_sum: RowSVector<f64, PARAMS_CNT> =
                nalgebra::zero();
    
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
        let direction = self.grad();

        direction.norm()
    }

    fn params(&self) -> &[f64] {
        &self.p
    }
}