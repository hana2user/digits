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
        ctx.beginPath();
        const pos = getCanvasCoordinates(e, canvas);
        ctx.moveTo(pos.x, pos.y);
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!drawing) return;
        const pos = getCanvasCoordinates(e, canvas);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    });

    canvas.addEventListener("mouseup", () => {
        drawing = false;
    });

    canvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        drawing = true;
        const touch = e.touches[0];
        const pos = getCanvasCoordinates(touch, canvas);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
      }, { passive: false });
      
      canvas.addEventListener("touchmove", (e) => {
        e.preventDefault();
        if (!drawing) return;
        const touch = e.touches[0];
        const pos = getCanvasCoordinates(touch, canvas);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
      }, { passive: false });
    
    canvas.addEventListener("touchend", () => {
        drawing = false;
    });

    canvas.addEventListener("touchstart", (e) => e.preventDefault(), { passive: false });
    canvas.addEventListener("touchmove", (e) => e.preventDefault(), { passive: false });

    loadModel();
});

function getCanvasCoordinates(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
  
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }
