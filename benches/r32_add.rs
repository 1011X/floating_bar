use criterion::{
	black_box,
	criterion_group,
	criterion_main,
	Criterion
};

use floating_bar::r32;

criterion_group!(
	benches, 
	i32_addition,
	f32_addition,
	r32_addition,
);
criterion_main!(benches);


fn i32_addition(c: &mut Criterion) {
	let mut ints = Vec::new();
	for i in 1_i32..=1000 {
		ints.push(i);
	}
	c.bench_function("i32_add 1000", |b| b.iter(|| {
		ints.iter().fold(black_box(0), |a, b| a + *b)
	}));
}

fn f32_addition(c: &mut Criterion) {
	let mut floats = Vec::new();
	for i in 1_u32..=1000 {
		let r = r32::new((i + 1000) as i32, i).unwrap();
		floats.push(f32::from(r));
	}
	c.bench_function("f32_add 1000", |b| b.iter(|| {
		floats.iter().fold(black_box(0.0), |a, b| a + *b)
	}));
}

fn r32_addition(c: &mut Criterion) {
	let mut ratios = Vec::new();
	for i in 1_u32..=1000 {
		ratios.push(r32::new((i + 1000) as i32, i).unwrap());
	}
	c.bench_function("r32_add 1000", |b| b.iter(|| {
		ratios.iter().fold(black_box(r32!(0)), |a, b| a + *b)
	}));
}
