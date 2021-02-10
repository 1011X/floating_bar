use criterion::{
	black_box,
	criterion_group,
	criterion_main,
	Criterion
};

use floating_bar::r32;

criterion_group!(
	benches, 
	f32_addition,
	r32_addition,
	f32_multiplication,
	r32_multiplication
);
criterion_main!(benches);

fn f32_addition(c: &mut Criterion) {
	let mut floats = Vec::new();
	for i in 0..1000 {
		floats.push(i as f32);
	}
	c.bench_function("f32_add 1000", |b| b.iter(|| {
		floats.iter().fold(black_box(0.0), |a, b| a + *b)
	}));
}

fn r32_addition(c: &mut Criterion) {
	let mut ratios = Vec::new();
	for i in 0_i16..1000 {
		ratios.push(r32::from(i));
	}
	c.bench_function("r32_add 1000", |b| b.iter(|| {
		ratios.iter().fold(black_box(r32::new(0, 1)), |a, b| a + *b)
	}));
}

fn f32_multiplication(c: &mut Criterion) {
	let mut floats = Vec::new();
	for i in 0_i16..1000 {
		floats.push(i as f32);
	}
	c.bench_function("f32_mult 1000", |b| b.iter(|| {
		floats.iter().fold(black_box(0.0), |a, &b| a * b)
	}));
}

fn r32_multiplication(c: &mut Criterion) {
	let mut ratios = Vec::new();
	for i in 0_i16..1000 {
		ratios.push(r32::from(i));
	}
	c.bench_function("r32_mult 1000", |b| b.iter(|| {
		ratios.iter().fold(black_box(r32::new(0, 1)), |a, &b| a * b)
	}));
}
