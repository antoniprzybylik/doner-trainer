use nalgebra::DVector;
use nalgebra::RowDVector;

use super::Network;

pub fn cost(
    y_values: &[DVector<f64>],
    d_values: &[DVector<f64>],
) -> f64 {
    let mut result: f64 = 0.;
    for (y, d) in y_values.iter().zip(d_values.iter()) {
        assert_eq!(y.len(), d.len());
        result += ((y - d).transpose() * (y - d)).trace();
    }

    result
}

pub fn apply_step(p: &mut [f64], step: &RowDVector<f64>) {
    assert_eq!(step.len(), p.len());

    for i in 0..p.len() {
        p[i] += step[i];
    }
}

pub fn revert_step(p: &mut [f64], step: &RowDVector<f64>) {
    assert_eq!(step.len(), p.len());

    for i in 0..p.len() {
        p[i] -= step[i];
    }
}

pub fn eval_untouched<N: Network>
    (p: &mut [f64], step: &RowDVector<f64>,
     x_values: &[DVector<f64>],
     d_values: &[DVector<f64>])
    -> f64
{
    assert_eq!(step.len(), N::PARAMS_CNT);

    apply_step(p, step);

    let mut y_values: Vec<DVector<f64>> =
        Vec::with_capacity(d_values.len());
    for x in x_values {
        assert_eq!(x.len(), N::NEURONS_IN);
        let x = x.clone();
        y_values.push(N::eval(&p, x));
    }

    revert_step(p, step);
    cost(y_values.as_slice(), d_values)
}

const P0: f64 = 0.000001f64;
const MAX_E: f64 = P0;

const PHI2: f64 = 2.618033988749894848207f64;
const RPHI: f64 = 0.618033988749894848207f64;

pub fn choose_step<N: Network>
    (p: &mut [f64],
     x_values: &[DVector<f64>],
     d_values: &[DVector<f64>],
     direction: RowDVector<f64>)
    -> RowDVector<f64>
{
    let (mut x1, mut x2, mut x3, mut x4): (f64, f64, f64, f64);
    let (fx1, mut fx3, mut fx4): (f64, f64, f64);

    let mut y_values: Vec<DVector<f64>> =
        Vec::with_capacity(d_values.len());
    for x in x_values {
        assert_eq!(x.len(), N::NEURONS_IN);
        let x = x.clone();
        y_values.push(N::eval(&p, x));
    }
    let y_values = y_values.as_slice();
    fx1 = cost(y_values, d_values);

    x1 = 0.;
    x2 = P0;
    while eval_untouched::<N>
        (p, &(x2*direction.clone()), x_values, d_values) <= fx1
    {
        x2 = x1 + (x2 - x1)*PHI2;
    }

	x3 = x2 - (x2 - x1)*RPHI;
	x4 = x1 + (x2 - x1)*RPHI;
	fx3 = eval_untouched::<N>
        (p, &(x3*direction.clone()), x_values, d_values);
	fx4 = eval_untouched::<N>
        (p, &(x4*direction.clone()), x_values, d_values);
	while (x1 - x2).abs() > MAX_E {
		if fx3 < fx4 {
			x2 = x4;

			fx4 = fx3;
			x3 = x2 - (x2 - x1)*RPHI;
			x4 = x1 + (x2 - x1)*RPHI;
			fx3 = eval_untouched::<N>
                (p, &(x3*direction.clone()), x_values, d_values);
		} else {
			x1 = x3;

			fx3 = fx4;
			x3 = x2 - (x2 - x1)*RPHI;
			x4 = x1 + (x2 - x1)*RPHI;
			fx4 = eval_untouched::<N>
                (p, &(x4*direction.clone()), x_values, d_values);
		}
	}

	((x1 + x2) / 2.0) * direction
}
