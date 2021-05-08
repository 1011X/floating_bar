use criterion::{
	black_box,
	criterion_group,
	criterion_main,
	Criterion
};

use floating_bar::r64;

criterion_group!(
	benches,
	i64_addition,
	f64_addition,
	r64_addition,
);
criterion_main!(benches);


fn i64_addition(c: &mut Criterion) {
	let mut ints = Vec::new();
	for i in 0_i64..1000 {
		ints.push(i);
	}
	c.bench_function("i64_add 1000", |b| b.iter(|| {
		ints.iter().fold(black_box(0), |a, b| a + *b)
	}));
}

fn f64_addition(c: &mut Criterion) {
	let mut floats = Vec::new();
	for i in 1_u64..=1000 {
		let r = r64::new((i + 1000) as i64, i).unwrap();
		floats.push(f64::from(r));
	}
	c.bench_function("f64_add 1000", |b| b.iter(|| {
		floats.iter().fold(black_box(0.0), |a, b| a + *b)
	}));
}

fn r64_addition(c: &mut Criterion) {
	let mut ratios = Vec::new();
	for i in 1_u64..=1000 {
		ratios.push(r64::new((i + 1000) as i64, i).unwrap());
	}
	c.bench_function("r64_add 1000", |b| b.iter(|| {
		ratios.iter().fold(black_box(r64!(0)), |a, b| a + *b)
	}));
}
