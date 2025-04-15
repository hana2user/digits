let model;
let canvas, ctx;

export function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

export async function predict() {
    const rgb = tf.browser.fromPixels(canvas);
    const gray = rgb.mean(2).expandDims(-1);

    const img = tf.scalar(1).sub(
        gray.resizeNearestNeighbor([28, 28])
            .toFloat()
            .div(255.0)
    ).expandDims(0);

    const data = await img.data();

    const previewCanvas = document.getElementById('preview');
    const previewCtx = previewCanvas.getContext('2d');
    const imageData = new ImageData(28, 28);
    for (let i = 0; i < 28 * 28; i++) {
        const val = data[i] * 255;
        imageData.data[i * 4 + 0] = val;
        imageData.data[i * 4 + 1] = val;
        imageData.data[i * 4 + 2] = val;
        imageData.data[i * 4 + 3] = 255;
    }
    previewCtx.putImageData(imageData, 0, 0);

    const prediction = model.predict(img);
    const result = prediction.argMax(1).dataSync()[0];
    document.getElementById('result').innerText = "Predicted: " + result;
}

async function loadModel() {
    model = await tf.loadLayersModel('tfjs_model/model.json');
    console.log("Model loaded");
}

window.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 15;
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";

    let drawing = false;

    canvas.addEventListener("mousedown", (e) => {
        drawing = true;
        const rect = canvas.getBoundingClientRect();
        ctx.beginPath();
        ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!drawing) return;
        const rect = canvas.getBoundingClientRect();
        ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
        ctx.stroke();
    });

    canvas.addEventListener("mouseup", () => {
        drawing = false;
    });

    loadModel();
});
