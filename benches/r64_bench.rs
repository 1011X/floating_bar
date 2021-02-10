use criterion::{
	black_box,
	criterion_group,
	criterion_main,
	Criterion
};

use floating_bar::r64;

criterion_group!(
	benches, 
	f64_addition,
	r64_addition,
	f64_multiplication,
	r64_multiplication
);
criterion_main!(benches);


fn f64_addition(c: &mut Criterion) {
	let mut floats = Vec::new();
	for i in 0..1000 {
		floats.push(i as f64);
	}
	c.bench_function("f64_add 1000", |b| b.iter(|| {
		floats.iter().fold(black_box(0.0), |a, b| a + *b)
	}));
}

fn r64_addition(c: &mut Criterion) {
	let mut ratios = Vec::new();
	for i in 0_i16..1000 {
		ratios.push(r64::from(i));
	}
	c.bench_function("r64_add 1000", |b| b.iter(|| {
		ratios.iter().fold(black_box(r64::new(0, 1)), |a, b| a + *b)
	}));
}

fn f64_multiplication(c: &mut Criterion) {
	let mut floats = Vec::new();
	for i in 0_i16..1000 {
		floats.push(i as f64);
	}
	c.bench_function("f64_mult 1000", |b| b.iter(|| {
		floats.iter().fold(black_box(0.0), |a, &b| a * b)
	}));
}

fn r64_multiplication(c: &mut Criterion) {
	let mut ratios = Vec::new();
	for i in 0_i16..1000 {
		ratios.push(r64::from(i));
	}
	c.bench_function("r64_mult 1000", |b| b.iter(|| {
		ratios.iter().fold(black_box(r64::new(0, 1)), |a, &b| a * b)
	}));
}
