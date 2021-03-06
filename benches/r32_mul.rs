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
	f32_multiplication,
	r32_multiplication
);
criterion_main!(benches);


fn i32_multiplication(c: &mut Criterion) {
	let mut ints = Vec::new();
	for i in 1_i32..=1000 {
		ints.push(i);
	}
	c.bench_function("i32_mult 1000", |b| b.iter(|| {
		ints.iter().fold(black_box(1), |a, &b| a * b)
	}));
}

fn f32_multiplication(c: &mut Criterion) {
	let mut floats = Vec::new();
	for i in 1_u32..=1000 {
		let r = r32::new((i + 1000) as i32, i).unwrap();
		floats.push(f32::from(r));
	}
	c.bench_function("f32_mult 1000", |b| b.iter(|| {
		floats.iter().fold(black_box(1.0), |a, &b| a * b)
	}));
}

fn r32_multiplication(c: &mut Criterion) {
	let mut ratios = Vec::new();
	for i in 1_u32..=1000 {
		ratios.push(r32::new((i + 1000) as i32, i).unwrap());
	}
	c.bench_function("r32_mult 1000", |b| b.iter(|| {
		ratios.iter().fold(black_box(r32!(1)), |a, &b| a * b)
	}));
}
