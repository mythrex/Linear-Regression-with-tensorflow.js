let xs = [];
let ys = [];
let theta0, theta1;

function setup() {
	createCanvas(400, 400);
	background(0);

	theta0 = tf.variable(tf.scalar(random(1)));
	theta1 = tf.variable(tf.scalar(random(1)));
}

// for normalizing the X and Y
function normalizeX(x) {
	return map(x, 0, width, 0, 1);
}
function normalizeY(y) {
	return map(y, 0, height, 1, 0);
}

function mousePressed() {
	let x = normalizeX(mouseX);
	let y = normalizeY(mouseY);
	xs.push(x);
	ys.push(y);
}
// function for deNormalizing
function deNormalizeX(x) {
	return map(x, 0, 1, 0, width);
}
function deNormalizeY(y) {
	return map(y, 0, 1, height, 0);
}

function draw() {
	background(0);
	stroke(255);
	strokeWeight(8);
	for (let i = 0; i < xs.length; i++) {
		let px = deNormalizeX(xs[i]);
		let py = deNormalizeY(ys[i]);
		point(px, py);
	}
}