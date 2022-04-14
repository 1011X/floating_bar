use criterion::{
	black_box,
	criterion_group,
	criterion_main,
	Criterion
};

use floating_bar::r32;

criterion_group!(
	benches,
	i32_division,
	f32_division,
	r32_division,
);
criterion_main!(benches);


fn i32_division(c: &mut Criterion) {
	let mut ints = Vec::new();
	for i in 1_i32..=7 {
        ints.push(i);
	}
	c.bench_function("i32_div 7!", |b| b.iter(|| {
		ints.iter().fold(black_box(5040), |a, b| a / *b)
	}));
}

fn f32_division(c: &mut Criterion) {
	let mut floats = Vec::new();
	for i in 1_u32..=7 {
// 		let r = r32::new((i + 1000) as i32, i).unwrap();
// 		floats.push(f32::from(r));
        floats.push(i as f32);
	}
	c.bench_function("f32_div 7!", |b| b.iter(|| {
		floats.iter().fold(black_box(5040.0), |a, b| a / *b)
	}));
}

fn r32_division(c: &mut Criterion) {
	let mut ratios = Vec::new();
	for i in 1_i32..=7 {
		ratios.push(r32::new(i, 1).unwrap());
	}
	c.bench_function("r32_div 7!", |b| b.iter(|| {
		ratios.iter().fold(black_box(r32!(5040)), |a, b| a / *b)
	}));
}
