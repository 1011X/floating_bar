use criterion::{
	black_box,
	criterion_group,
	criterion_main,
	Criterion
};

use floating_bar::r32;

criterion_group!(
	benches, 
	i32_multiplication,
	r32_multiplication
);
criterion_main!(benches);


fn i32_multiplication(c: &mut Criterion) {
	let mut ints = Vec::new();
	for i in 0_i32..1000 {
		ints.push(i);
	}
	c.bench_function("i32_mult 1000", |b| b.iter(|| {
		ints.iter().fold(black_box(1), |a, &b| a * b)
	}));
}

fn r32_multiplication(c: &mut Criterion) {
	let mut ratios = Vec::new();
	for i in 0_i16..1000 {
		ratios.push(r32::from(i));
	}
	c.bench_function("r32_mult 1000", |b| b.iter(|| {
		ratios.iter().fold(black_box(r32!(1)), |a, &b| a * b)
	}));
}
