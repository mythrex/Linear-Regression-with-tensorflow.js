let xs = [];
let ys = [];
let theta0, theta1;

// using gradient decent
const learningRate = 0.02;
const optimizer = tf.train.sgd(learningRate);

// loss function
const loss = (pred, label) =>
	pred
		.sub(label)
		.square()
		.mean();

function setup() {
	createCanvas(400, 400);
	background(0);

	theta0 = tf.variable(tf.scalar(random(1)));
	theta1 = tf.variable(tf.scalar(random(1)));
}

function predict(xs) {
	const tfxs = tf.tensor1d(xs);
	// y = theta1 * x + theta0
	const ys = tfxs.mul(theta1).add(theta0);
	return ys;
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

	if (xs.length > 0) {
		// change the weight
		const tfys = tf.tensor1d(ys);
		optimizer.minimize(() => loss(predict(xs), tfys));
	}

	let x_line = [0, 1];
	let y_line = predict(x_line);
	let x1 = deNormalizeX(x_line[0]);
	let x2 = deNormalizeX(x_line[1]);

	y_line = y_line.dataSync();
	let y1 = deNormalizeY(y_line[0]);
	let y2 = deNormalizeY(y_line[1]);

	line(x1, y1, x2, y2);
}
