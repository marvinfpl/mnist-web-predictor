function drawOnCanvas() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let draw = false;

    canvas.addEventListener("mousedown", (e) => {
        draw = true;
        ctx.beginPath();
        ctx.moveTo(e.offsetX, e.offsetY);
    });

    canvas.addEventListener("mouseup", () => {
        draw = false;
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!draw) return;

        ctx.lineCap = "round";
        ctx.lineWidth = 22;
        ctx.strokeStyle = "Black";

        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    });
}

function clearCanvas() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function predictDigit() {
    const canvas = document.getElementById("canvas");
    const image = canvas.toDataURL("image/png");

    clearCanvas();

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({image})
    })
    .then(res => res.json())
    .then(data => {
        console.log(data)
        document.getElementById("prediction").innerHTML = `<strong>Prediction: </strong> ${data.prediction} with a solid ${data.confidence_score}% confidence score! `;
    })
    .catch(err => console.error(err)
    );
}